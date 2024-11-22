import json
import time
import warnings
from typing import Dict, List, Generator, Optional

import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import notebook as tqdm

from mycode.multi_ctgan import MultiTableCTGAN
from mycode.testcode.metabuilder import MetaBuilder
from mycode.sdv.evaluation import evaluate
from mycode.test_20_tables import build_sdv_metadata_from_origin_tables
from mycode.testcode import Xargs
from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.base import DataProcessor
from sdgx.data_processors.manager import DataProcessorManager
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.utils import logger

CATEGORY_THRESHOLD = 15


def _is_categorical(column: pd.Series | pd.DataFrame) -> bool:
    return int(column.nunique()) <= CATEGORY_THRESHOLD


class RFECV_CTGANTrainingData:
    def __init__(self,
                 table_name: str,
                 attached_columns: Optional[List[str]] = None,  # 因为processed后的数据有可能直接被删了，没法rfecv。
                 target_column_name_selected_map: Optional[Dict[str, List[str]]] = None,
                 tables: Optional[pd.DataFrame] = None,
                 origin_columns: Optional[List[str]] = None,
                 relative_columns: Optional[List[str]] = None,
                 trained_columns: Optional[List[str]] = None,
                 ):
        self.need_fitted = origin_columns is not None
        self.tables = tables
        self.origin_columns = origin_columns or []
        self.relative_columns = relative_columns or []
        self.target_column_name_selected_map = target_column_name_selected_map or {}
        self.trained_columns = trained_columns or []
        self.attached_columns = attached_columns or []
        self.table_name = table_name
        self.fitted_model: Optional[CTGANSynthesizerModel] = None

    def set_fitted_model(self, fitted_model):
        self.fitted_model = fitted_model


class MultiTableRfecvCTGAN:
    def __init__(self, db_path, x_args: Xargs.XArg, temp_name="testrfecv", exclude_processor=[]):
        self.metadata: Optional[Metadata] = None
        self.multi_x_join = MultiTableCTGAN(db_path=db_path, x_args=x_args, temp_name=temp_name)
        self.meta_builder: Optional[MetaBuilder] = None
        self.to_training: List[RFECV_CTGANTrainingData] = []
        self.data_processors_manager = DataProcessorManager()
        data_processors = [item for item in self.data_processors_manager.registed_default_processor_list if
                           item not in exclude_processor]
        logger.info(f"Using data processors: {data_processors}")
        self.data_processors = [
            (
                d if isinstance(d, DataProcessor) else (
                    self.data_processors_manager.init_data_processor(d)
                )
            )
            for d in data_processors
        ]

    def init(self, add_cols_num=0, added_data_path=False, meta_builder: MetaBuilder = None):
        self.multi_x_join.init(add_cols_num=add_cols_num, added_data_path=added_data_path)
        self.meta_builder = meta_builder

        def generator():
            yield self.multi_x_join.join_tables.copy()

        data_connector = GeneratorConnector(generator)
        data_loader = DataLoader(data_connector)
        self.metadata = Metadata.from_dataloader(data_loader)
        if self.meta_builder:
            self.metadata = self.meta_builder.build(multi_wrapper=self.multi_x_join, metadata=self.metadata)
        preprocessed_loader = self._data_preprocess(data_loader, metadata=self.metadata)

        # TODO 优化，这里调用了loadall两次 ？？ ，一次在loader的__init__时候
        preprocessed_data = preprocessed_loader.load_all()

        assert not preprocessed_data.isna().any().any()
        assert not preprocessed_data.isnull().any().any()

        attached_columns_data = None
        if set(preprocessed_data.columns) != set(self.multi_x_join.join_tables):
            warnings.warn("Detected columns deleting by data processors, we will attach original data to train RFECV.")
            missing_columns = set(self.multi_x_join.join_tables) - set(preprocessed_data.columns)
            attached_columns_data = pd.DataFrame(self.multi_x_join.join_tables[list(missing_columns)])

        self.to_training = MultiTableRfecvCTGAN._argument_tables(
            join_tables=preprocessed_data,  # TODO 优化
            attached_columns_data=attached_columns_data,
            tables_name=self.multi_x_join.origin_tables.keys(),
            columns_table_map=self.multi_x_join.join_columns_table_name_map,
            join_columns_map=self.multi_x_join.join_columns_map,
            metadata=self.metadata)

        preprocessed_loader.finalize(clear_cache=True)

    def _data_preprocess(self, dataloader: DataLoader, metadata):
        start_time = time.time()
        for d in self.data_processors:
            if dataloader:
                d.fit(metadata=metadata, tabular_data=dataloader)
            else:
                d.fit(metadata=metadata)
        logger.info(
            f"Fitted {len(self.data_processors)} data processors in  {time.time() - start_time}s."
        )

        def chunk_generator() -> Generator[pd.DataFrame, None, None]:
            for chunk in dataloader.iter():
                for d in self.data_processors:
                    chunk = d.convert(chunk)

                assert not chunk.isna().any().any()
                assert not chunk.isnull().any().any()
                yield chunk

        logger.info("Initializing processed data loader...")
        start_time = time.time()
        processed_dataloader = DataLoader(
            GeneratorConnector(chunk_generator),
            identity=dataloader.identity
        )
        logger.info(f"Initialized processed data loader in {time.time() - start_time}s")

        # TODO 优化 Onehot转换 到此处
        # discrete_columns = list(metadata.get("discrete_columns"))
        # transformer = DataTransformer(metadata=metadata)
        # logger.info("Fitting model's transformer...")
        # self._transformer.fit(dataloader, discrete_columns)
        # logger.info("Transforming data...")
        # self._ndarry_loader = self._transformer.transform(dataloader)
        # TODO 还需要考虑CTGAN后续的训练。

        return processed_dataloader

    @staticmethod
    def _convert_categorical(data: pd.DataFrame, metadata: Metadata, target_column_name: str):
        discrete_columns = set(metadata.discrete_columns)
        onehot_columns_map = {}
        for col in discrete_columns:
            if col not in data:
                continue
            if target_column_name != col and _is_categorical(data[[col]]):
                # Onehot Encode
                onehot_data = pd.get_dummies(data[[col]], prefix=col, prefix_sep="_Onehot_")
                data = data.drop(col, axis=1)
                data = pd.concat([data, onehot_data], axis=1)
                onehot_columns_map.update({
                    c: col for c in onehot_data.columns
                })
            else:
                # Label encode
                mapper = {v: k for k, v in enumerate(sorted(pd.unique(data[col]), key=lambda x: x))}
                data[col] = data[col].map(mapper)
        return onehot_columns_map, discrete_columns, data

    @staticmethod
    def _convert_nan(data: pd.DataFrame, metadata: Metadata, target_column_name: str):
        filled_value = 'NAN_VALUE'
        if target_column_name in metadata.int_columns:
            filled_value = 0
        elif target_column_name in metadata.float_columns:
            filled_value = 0.0
        elif target_column_name in metadata.datetime_columns:
            filled_value = data[target_column_name].mean()
        data[target_column_name] = data[[target_column_name]].fillna(filled_value)
        return data
        # # int columns
        # int_columns = set(metadata.int_columns)
        # # float columns
        # float_columns = set(metadata.float_columns)
        #
        # # fill numeric nan value
        # for col in int_columns:
        #     data[col] = data[col].fillna(fill_na_value_int)
        # for col in float_columns:
        #     data[col] = data[col].fillna(fill_na_value_float)
        #
        # # fill other non-numeric nan value
        # for col in data.columns:
        #     if col in int_columns or col in float_columns:
        #         continue
        #     data[col] = data[col].fillna(fill_na_value_default)

    @staticmethod
    def _do_rfecv(target_column_name: str, data: pd.DataFrame, metadata: Metadata) -> set[str]:
        # data = data.copy()

        # TODO 在这里添加处理NaN值。这里应该不需要，因为在上面的process处理了，但是现在有bug
        # data = MultiTableRfecvCTGAN._convert_nan(data, metadata, target_column_name)

        # TODO 这是临时使用的onehot，每次会重复很多次。考虑优化
        onehot_columns_map, discrete_columns, data = MultiTableRfecvCTGAN._convert_categorical(data, metadata,
                                                                                               target_column_name)

        assert not data[[target_column_name]].isnull().any().any()
        assert not data[[target_column_name]].isna().any().any()

        # RFECV
        X = data.drop(target_column_name, axis=1)
        y = data[[target_column_name]]
        if y.shape[0] == 0 or y.shape[1] == 0 or X.shape[1] == 1:
            raise ValueError('数据为空')

        if target_column_name in discrete_columns and _is_categorical(y):
            estimator = DecisionTreeClassifier(random_state=0)
        else:
            estimator = DecisionTreeRegressor(random_state=0)
        rfecv = RFECV(estimator, min_features_to_select=3)
        rfecv.fit(X, y)

        # 解析相关列，排除 Onehot
        feature_ranks = rfecv.ranking_
        feature_names = X.columns.tolist()
        sorted_features = sorted(zip(feature_names, feature_ranks), key=lambda x: x[1])

        encoded_k = int(rfecv.n_features_)
        encoded_features = [name for name, _ in sorted_features][:encoded_k]
        # encoded_scores = rfecv.cv_results_["mean_test_score"].tolist()
        # encoded_scores_std = rfecv.cv_results_["std_test_score"].tolist()
        decoded_features: list[str] = [
            onehot_columns_map[feature] if feature in onehot_columns_map else feature
            for feature in encoded_features
        ]
        decoded_k = len(decoded_features)
        # decoded_scores: list[float] = []
        # decoded_scores_std: list[float] = []
        return set(decoded_features[:decoded_k])

    @staticmethod
    def _argument_tables(*, join_tables: pd.DataFrame,
                         attached_columns_data: pd.DataFrame = None,
                         tables_name: List[str],
                         columns_table_map: Dict[str, List[str]],
                         join_columns_map: Dict[str, Dict[str, str]],
                         metadata: Metadata) -> List:
        tablekeyd_convert_to_origin_columns_map = {
            key: {v: k for k, v in value.items()} for key, value in join_columns_map.items()
        }

        if attached_columns_data is None:
            attached_columns_data = pd.DataFrame()
        for tbn in tablekeyd_convert_to_origin_columns_map:
            assert set(tablekeyd_convert_to_origin_columns_map[tbn].values()).issubset(
                set(join_tables.columns).union(attached_columns_data.columns))
        attached_tables = pd.concat([attached_columns_data, join_tables], axis=1)
        attached_columns = set(attached_columns_data.columns)

        selected_columns = set()
        to_training = []

        # testset = frozenset(join_tables.columns)

        psb = tqdm.tqdm(desc="Choosing Cols", total=len(join_tables.columns))
        # 对每一个表的每一列 和 未被选择的列 进行 RFECV
        for table_name in tables_name:

            # assert testset == frozenset(join_tables.columns)
            # 其他表的非选择的列
            other_table_names = set(tables_name) - {table_name}
            all_columns_name = set(join_tables.columns) - selected_columns
            other_table_columns_name = set(
                cn for cn in all_columns_name
                if other_table_names.intersection(set(columns_table_map[cn])))

            current_table_columns: list[str] = list(tablekeyd_convert_to_origin_columns_map[table_name].values())

            # assert not (set(current_table_columns) - set(join_tables.columns))
            # assert set(current_table_columns).issubset(set(join_tables.columns))

            if not set(current_table_columns) - selected_columns:
                to_training.append(RFECV_CTGANTrainingData(
                    table_name=table_name,
                ))
                continue
            # 对每个表的【每一列】
            target_column_name_selected_map: Dict[str, List[str]] = {}
            to_training_columns = set(current_table_columns)
            for target_column_name in current_table_columns:
                psb.update(1)
                # 如果是参考列，只算全大表第一次的RFECV，后续选择不再考虑，但还要跟随本表其他列进行计算RFECV。
                # 如果是独特列，同理。
                # if MultiTableCTGAN.is_ref_column(target_column_name):
                if target_column_name in selected_columns or target_column_name in attached_columns:
                    continue
                to_rfe_columns_name = set(current_table_columns).union(other_table_columns_name)
                try:
                    to_rfe_columns = join_tables[list(to_rfe_columns_name)].copy()
                    # 对被processor删掉的列（如email）进行REFCV，没有意义。还会吧混杂在里面的None和Nan加入到已经清洗的数据里。
                except KeyError:
                    to_rfe_columns = attached_tables[list(to_rfe_columns_name)].copy()

                # rfe
                current_selected_columns = set(
                    MultiTableRfecvCTGAN._do_rfecv(target_column_name, to_rfe_columns, metadata)
                )

                # current_selected_columns = random.choices(list(to_rfe_columns_name), k=3)

                selected_columns = selected_columns.union(current_selected_columns)
                selected_columns.add(target_column_name)
                to_training_columns = to_training_columns.union(current_selected_columns)
                target_column_name_selected_map.update({
                    target_column_name: list(current_selected_columns)
                })
            # 每个表生成一个模型
            current_attached_columns = set(attached_columns_data.columns).intersection(to_training_columns)

            to_training_tables = join_tables[list(to_training_columns - current_attached_columns)].copy()

            assert not to_training_tables.isnull().any().any()
            assert not to_training_tables.isna().any().any()

            relative_columns = []
            for rset in target_column_name_selected_map.values():
                relative_columns.extend(rset)
            to_training.append(RFECV_CTGANTrainingData(
                table_name=table_name,
                attached_columns=list(current_attached_columns),
                target_column_name_selected_map=target_column_name_selected_map,
                tables=to_training_tables,
                origin_columns=list(current_table_columns),
                relative_columns=list(set(relative_columns) - set(current_table_columns)),
                # 防止选本表的  # 后面的列只从这里取出来，后面生成的不要 TODO 并行化时注意优先级
                trained_columns=list(to_training_columns)
            ))
        return to_training

    def _fit_argument_table_ctgan(self, to_training_info: RFECV_CTGANTrainingData, *, epochs=1, batch_size=500,
                                  device="cpu"):
        logger.info("Fitting RFE-CTGAN model on table {}...".format(to_training_info.table_name))

        assert not to_training_info.tables.isnull().any().any()
        assert not to_training_info.tables.isna().any().any()

        def partition_generator():
            yield to_training_info.tables.copy()

        partition_data_connector = GeneratorConnector(partition_generator)
        partition_data_loader = DataLoader(partition_data_connector)

        # metadata 从全局的提取，不要新建，不然已经转换完类型存在错误
        metadata_g: dict = json.loads(self.metadata._dump_json())

        def do_remove_columns(key, obj=metadata_g, exists=to_training_info.trained_columns):
            if isinstance(obj[key], list):
                obj[key] = [item for item in obj[key] if item in exists]
            elif isinstance(obj[key], dict):
                obj[key] = {
                    k: v for k, v in obj[key].items()
                    if k in exists
                }
            else:
                raise Exception()

        to_escape_remove_keys = ["version", "numeric_format"]
        for item in metadata_g.keys():
            if item in to_escape_remove_keys:
                continue
            do_remove_columns(item)
        for k in ["positive", "negative"]:
            do_remove_columns(k, metadata_g["numeric_format"])
        partition_metadata = Metadata.load_json(metadata_g)
        partition_metadata.check()
        # partition_metadata = Metadata.from_dataloader(partition_data_loader)
        # if self.meta_builder:
        #     partition_metadata = self.meta_builder.build(multi_wrapper=None, metadata=partition_metadata)

        # TODO 这里的onehot可以考虑修改移到上面去，但是要解决返回为pd.DF而不是np.array
        # discrete_columns = list(metadata.get("discrete_columns"))
        # transformer = DataTransformer(metadata=metadata)
        # logger.info("Fitting model's transformer...")
        # self._transformer.fit(dataloader, discrete_columns)
        # logger.info("Transforming data...")
        # self._ndarry_loader = self._transformer.transform(dataloader)

        # TODO 优化：现在ctgan给什么列就会训练输出什么列，但是现在对于某些列（前面子表已经选择的列），只需要作为输入，不需要输出。
        partition_ctgan = CTGANSynthesizerModel(
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )
        partition_ctgan.fit(metadata=partition_metadata, dataloader=partition_data_loader)
        to_training_info.fitted_model = partition_ctgan

    def fit(self, *, epochs=1, batch_size=500, device="cpu"):
        # 对每一个 训练对象 训练一个ctgan

        for i, v in enumerate(self.to_training):
            if not v.need_fitted:
                continue
            st = time.time()
            self._fit_argument_table_ctgan(v, epochs=epochs, batch_size=batch_size, device=device)
            ed = time.time()
            print(f"data_dim: {v.fitted_model.data_dim}, time: {round(ed - st, 2)}s")
        return self

    def _sample_one_table(self, count, model) -> pd.DataFrame:
        missing_count = count
        max_trails = 5
        sample_data_list = []
        psb = tqdm.tqdm(total=count, desc="Sampling")
        while missing_count > 0 and max_trails > 0:
            sample_data = model.sample(max(int(missing_count * 1.2), model._batch_size), drop_more=False)
            # 这里需要忽略错误（列不存在），因为processor是根据整个表训练的
            for d in self.data_processors:
                sample_data = d.reverse_convert(sample_data)
            # TODO attached列会重复出现（email），每次reverse都会生成一次。
            # TODO 这里面有Nan
            sample_data_list.append(sample_data)
            missing_count = missing_count - len(sample_data)
            psb.update(len(sample_data))
            max_trails -= 1
        res = pd.concat(sample_data_list)[:count]
        assert len(res) == count
        # TODO 这里有bug
        return res.reset_index(drop=True)

    def sample(self, n) -> Dict[str, pd.DataFrame]:
        # TODO 外键约束被破坏？join之后恢复？
        keyed_columns_convert_map = self.multi_x_join.join_columns_map
        tables_name = self.multi_x_join.origin_tables.keys()
        columns_table_name_map = self.multi_x_join.join_columns_table_name_map
        synthesized_tables: Dict[str: List[pd.DataFrame]] = {
            tbn: [] for tbn in tables_name
        }
        for i, v in enumerate(self.to_training):
            if not v.need_fitted:
                continue
            data = self._sample_one_table(n, v.fitted_model)
            columns = list(set(v.origin_columns).union(v.relative_columns).union(v.attached_columns))
            data = data[columns]
            # 提取当前表列
            real_current_columns = keyed_columns_convert_map[v.table_name].keys()
            # 包含了是本表的attach_columns，因此无需再下面的计算
            # delta_columns = set(v.attached_columns) - set(real_current_columns)
            # if delta_columns:
            exists_columns = synthesized_tables[v.table_name].columns
            # 要除去已经存在的列，他们是优先级更高的，按理说之前应该已经被data[columns]过滤了，这里再做个检验。(不对)
            # assert not list(set(exists_columns) & set(real_current_columns))
            # REF在后面代码会被附到每个生成表中，这里有可能重复生成了，只取前面生成的。
            # assert np.all(pd.Series(exists_columns).apply(self.multi_x_join.is_ref_column))
            fil_columns = list(set(real_current_columns) - set(exists_columns))
            synthesized_tables[v.table_name] = pd.concat([
                synthesized_tables[v.table_name],
                data[fil_columns]
            ], axis=1)

            # 提取其他表列，当前表列有可能是REF（其他表的），因此不能直接用relative
            reference_columns = {col for col in columns if self.multi_x_join.is_ref_column(col)}
            other_columns = set(v.relative_columns)
            to_add_columns = reference_columns.union(other_columns)
            # TODO pd.concat只使用一次，一次性使用pd.concat()比迭代concat更快。
            for col in to_add_columns:  # 寻找匹配的表名
                to_add_table_set = set(columns_table_name_map[col]) - {v.table_name}

                # assert col not in synthesized_tables[v.table_name].columns

                for to_add_table_name in to_add_table_set:
                    if to_add_table_name == v.table_name:
                        continue
                    # assert col not in synthesized_tables[to_add_table_name]
                    if col in synthesized_tables[to_add_table_name].columns:
                        # TODO REF有重复，只取前面的REF。影响评估结果，但保证Join
                        continue
                    synthesized_tables[to_add_table_name] = pd.concat([
                        synthesized_tables[to_add_table_name],
                        data[[col]]
                    ], axis=1)

        # if save:
        #     self.save_xlsx(synthesized_tables, f"{self.TABLE_TEMP_NAME}_sample_split.xlsx")
        #     self.save_xlsx(self.origin_tables, f"{self.TABLE_TEMP_NAME}_origin_split.xlsx")

        # 恢复列名，重新排序
        for key in synthesized_tables:
            synthesized_tables[key] = (
                synthesized_tables[key]
                .rename(
                    columns=keyed_columns_convert_map[key]
                ).reindex(
                    columns=self.multi_x_join.tablekeyed_columns_order[key]
                )
            )

        return synthesized_tables

    def evaluate(self, x_args, synthetic_data, prompt=""):
        real_ctgan_data = self.multi_x_join.origin_tables
        metadatasdv = build_sdv_metadata_from_origin_tables(real_ctgan_data, x_args, path=self.multi_x_join.db_path)
        # st = time.time()
        ret = evaluate(synthetic_data, real_ctgan_data, metadatasdv, aggregate=False)
        # ed = time.time()
        print(ret.normalized_score.mean())
        # ret
        ks, cs = float(ret[ret['metric'] == 'KSComplement']["raw_score"].iloc[0]), float(
            ret[ret['metric'] == 'CSTest']["raw_score"].iloc[0])
        #
        # self.TIME_LOGGER.write(
        #     f"eva {prompt} mean: {round(ret.normalized_score.mean(), 5)}, ks: {round(ks, 5)},cs: {round(cs, 5)}, evatime: {round(ed - st, 2)}, ")
        #
        # self.TIME_LOGGER.flush()
        return ks, cs
