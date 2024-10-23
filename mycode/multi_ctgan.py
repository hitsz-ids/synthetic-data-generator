import pickle
import random
from typing import List

import pandas as pd
from pandas import DataFrame

from mycode.test_20_tables import fetch_data_from_sqlite
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
    TABLE_TEMP_NAME = "test_100k"
    SEPERATOR = "_TABLE_"

    def __init__(self, *, db_path: str = './mycode/data_sqlite.db', temp_name="test_100k", x_table: List[str],
                 x_key: List[str], x_how: List[str]):
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
        self.sdv_metadata_, tables = fetch_data_from_sqlite(path=db_path)
        self.tables_ = tables
        self.origin_tables = {
            key: tables[key] for key in x_table
        }
        self.join_tables = None
        self.join_columns_map = {}

    def init(self, add_cols_num=0):
        """
        整合表格为一张表
        """

        if add_cols_num > 0:
            for i, table_name in enumerate(self.x_table):
                current_table = self.origin_tables[table_name]
                table_columns = list(current_table.columns)
                for j in range(add_cols_num):
                    col = random.choice(table_columns)
                    current_table[f'COPY{j}{col}'] = current_table[col]

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
                    self.join_tables = pd.merge(self.join_tables, current_table, on=on_key, how=self.x_how[i], suffixes=(False, False))
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

    def save_init_data(self, path: str):
        pickle.dump(self, open(path, "wb"))

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
        self.synthesizer.fit()
        print(f"data_dim: {self.ctgan.data_dim}")
        return self

    def sample(self, n, *, save=True) -> DataFrame:
        join_samples: DataFrame = self.synthesizer.sample(n)
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
