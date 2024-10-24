import pickle
import random
import time
from typing import List

import pandas as pd
from pandas import DataFrame
from mycode.sdv.evaluation import evaluate
from mycode.test_20_tables import fetch_data_from_sqlite, build_sdv_metadata_from_origin_tables
from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer


class MetaBuilder:
    def __init__(self):
        pass

    def build(self, multi_wrapper, metadata):
        raise NotImplementedError


class MultiTableCTGAN:
    SEPERATOR = "_TABLE_"
    def __del__(self):
        self.TIME_LOGGER.close()

    def __init__(self, *, db_path: str = './mycode/data_sqlite.db', temp_name="test_100k", x_table: List[str],
                 x_key: List[str], x_how: List[str]):
        self.TIME_LOGGER = open("./mycode/testcode/timelog.txt", "a+")
        self.TABLE_TEMP_NAME = "test_100k"


        self.db_path = db_path
        self.tables_ = None
        self.sdv_metadata_ = None
        self.synthesizer = None
        self.ctgan = None
        self.metadata = None
        self.data_loader = None
        self.data_connector = None
        self.x_table = x_table
        self.x_key = [None]
        self.x_key.extend(x_key)
        temp = list(x_how)
        temp.insert(0, None)
        self.x_how = temp
        self.TABLE_TEMP_NAME = temp_name
        self.origin_tables = {}
        self.join_tables = None
        self.join_columns_map = {}

    def init(self, add_cols_num=0, added_data_path=False):
        """
        整合表格为一张表
        """

        # fetch_data_from_sqlite_filterx(["BookLoan",
        #                                "Book", "Library", "Student",
        #                                "Enrollment", "Submission", "Course",  # "Assignment"
        #                                "CourseTextbook", "Textbook",
        #                                "Schedule", "Professor"], "1k_data_sqlite.db")

        if not added_data_path:
            self.sdv_metadata_, tables = fetch_data_from_sqlite(path=self.db_path)
            self.tables_ = tables
            self.origin_tables = {
                key: tables[key] for key in self.x_table
            }

        if add_cols_num > 0 and not added_data_path:
            for i, table_name in enumerate(self.x_table):
                current_table = self.origin_tables[table_name]
                table_columns = list(current_table.columns)
                added_cols = []
                for j in range(add_cols_num):
                    col = random.choice(table_columns)
                    added_cols.append(col)
                    current_table[f'{col}_COPY{j}'] = current_table[col]
                print(
                    f'{table_name} added: {" ".join(added_cols)}'
                )
        elif add_cols_num == 0 and added_data_path:
            with open(added_data_path, 'rb') as f:
                self.origin_tables = pickle.load(f)
        else:
            raise ValueError("不能同时指定 add_path 和 add_cols ")

        join_keys = set(self.x_key) - {None}
        for i, table_name in enumerate(self.x_table):
            current_table = self.origin_tables[table_name]
            table_columns = set(current_table.columns)
            self.join_columns_map[table_name] = {}
            unique_columns = table_columns - join_keys
            reference_columns = join_keys.intersection(table_columns)
            if unique_columns:
                convert_columns = {x: table_name + self.SEPERATOR + x for x in unique_columns}
                current_table = current_table.rename(columns=convert_columns)
                self.join_columns_map[table_name].update({
                    value: key for key, value in convert_columns.items()
                })
            if reference_columns:
                convert_columns = {x: "REF" + self.SEPERATOR + x for x in reference_columns}
                current_table = current_table.rename(columns=convert_columns)
                self.join_columns_map[table_name].update({
                    value: key for key, value in convert_columns.items()
                })
            # x_table_name
            if self.join_tables is not None:
                on_key = "REF" + self.SEPERATOR + self.x_key[i]
                try:
                    self.join_tables = pd.merge(self.join_tables, current_table, on=on_key, how=self.x_how[i],
                                                suffixes=(False, False))
                except ValueError as e:
                    print(f"\n{on_key=}")
                    # print(self.join_tables)
                    # print(current_table)
                    e.args = (e.args[0], [self.join_tables, current_table])
                    raise e
            else:
                self.join_tables = current_table
            print(f"{len(self.join_tables)}", end=" ")
        print("")
        print(f"{len(self.join_tables)} lines x {len(self.join_tables.columns)} columns")
        return self

    def save_add_data(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.origin_tables, f)

    def build_metadata(self, *, builder: MetaBuilder = None):
        """
            构建metadata
        """
        dataset_csv = f"{self.TABLE_TEMP_NAME}_original_joined.csv"
        self.join_tables.to_csv(dataset_csv, index=False)
        self.data_connector = CsvConnector(path=dataset_csv)
        self.data_loader = DataLoader(self.data_connector)
        self.metadata = Metadata.from_dataloader(self.data_loader)
        if builder:
            self.metadata = builder.build(multi_wrapper=self, metadata=self.metadata)
        return self

    def fit(self, *, epochs=1, batch_size=500, device="cpu"):
        self.ctgan = CTGANSynthesizerModel(
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )
        self.synthesizer = Synthesizer(
            metadata=self.metadata,
            model=self.ctgan,
            data_connector=self.data_connector,
        )
        st = time.time()
        self.synthesizer.fit()
        ed = time.time()
        print(f"data_dim: {self.ctgan.data_dim}, time: {round(ed - st, 2)}s")
        print(f"{len(self.join_tables)} lines x {len(self.join_tables.columns)} columns")

        self.TIME_LOGGER.write(f"\n{self.TABLE_TEMP_NAME}: {len(self.join_tables)} lines, {len(self.join_tables.columns)} columns, {self.ctgan.data_dim} dims, {round(ed - st, 2)}s train time, ")
        self.TIME_LOGGER.flush()
        return self

    def sample(self, n, *, save=True) -> DataFrame:
        st = time.time()
        join_samples: DataFrame = self.synthesizer.sample(n)
        ed = time.time()
        print("time: {:.2f}".format(ed - st))
        self.TIME_LOGGER.write(
            f"{round(ed - st, 2)}s sample {n} time, ")

        self.TIME_LOGGER.flush()
        join_samples.to_csv(f"{self.TABLE_TEMP_NAME}_sample_joined.csv", index=False)
        synthesized_tables = {
            table_name: join_samples[table_map.keys()].rename(columns=table_map) for table_name, table_map in
            self.join_columns_map.items()
        }
        if save:
            self.save_xlsx(synthesized_tables, f"{self.TABLE_TEMP_NAME}_sample_split.xlsx")
            self.save_xlsx(self.origin_tables, f"{self.TABLE_TEMP_NAME}_origin_split.xlsx")
        return synthesized_tables

    @staticmethod
    def save_xlsx(tables, name):
        # 使用ExcelWriter保存为多sheet的Excel文件
        with pd.ExcelWriter(name) as writer:
            for sheet_name, df in tables.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def read_xlsx(name):
        # 读取多sheet的Excel文件
        sheets_dict = pd.read_excel(name, sheet_name=None)
        return sheets_dict



    def evaluate(self, x_args, synthetic_ctgan_data, prompt=""):
        real_ctgan_data = self.origin_tables
        metadatasdv = build_sdv_metadata_from_origin_tables(real_ctgan_data, x_args, path=self.db_path)
        # print(metadatasdv)
        st = time.time()
        ret = evaluate(synthetic_ctgan_data, real_ctgan_data, metadatasdv, aggregate=False)
        ed = time.time()
        print(ret.normalized_score.mean())
        # ret
        ks, cs = float(ret[ret['metric']=='KSComplement']["raw_score"].iloc[0]), float(ret[ret['metric']=='CSTest']["raw_score"].iloc[0])

        self.TIME_LOGGER.write(
            f"eva {prompt} mean: {round(ret.normalized_score.mean(), 5)}, ks: {round(ks, 5)},cs: {round(cs, 5)}, evatime: {round(ed - st, 2)}, ")

        self.TIME_LOGGER.flush()