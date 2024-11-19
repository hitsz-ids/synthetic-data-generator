import time
import warnings
from typing import Dict, List, Generator, Optional
from pydantic import Field
import pandas as pd
from pydantic.dataclasses import dataclass
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.base import DataProcessor
from sdgx.data_processors.manager import DataProcessorManager
from sdgx.utils import logger
from mycode.multi_ctgan import MultiTableCTGAN, MetaBuilder
from mycode.testcode import Xargs

CATEGORY_THRESHOLD = 15


class RFECV_CTGANTrainingData:
    def __init__(self,
                 table_name: str,
                 attached_columns: Optional[List[str]] = None,
                 target_column_name_selected_map: Optional[Dict[str, List[str]]] = None,
                 tables: Optional[pd.DataFrame] = None,
                 columns: Optional[List[str]] = None
    ):
        self.tables = tables
        self.columns = columns or []
        self.target_column_name_selected_map = target_column_name_selected_map or {}
        self.attached_columns = attached_columns or []
        self.table_name = table_name



class MultiTableRfecvCTGAN:
    def __init__(self, db_path, x_args: Xargs.XArg, temp_name="testrfecv"):
        self.metadata = None
        self.multi_x_join = MultiTableCTGAN(db_path=db_path, x_args=x_args, temp_name=temp_name)
        self.meta_builder = None
        self.to_training: List[RFECV_CTGANTrainingData] = []

        self.data_processors_manager = DataProcessorManager()
        data_processors = self.data_processors_manager.registed_default_processor_list
        logger.info(f"Using data processors: {data_processors}")
        self.data_processors = [
            (
                d if isinstance(d, DataProcessor) else self.data_processors_manager.init_data_processor(d)
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
        metadata = Metadata.from_dataloader(data_loader)
        if self.meta_builder:
            self.metadata = self.meta_builder.build(multi_wrapper=self.multi_x_join, metadata=metadata)
        preprocessed_loader = self._data_preprocess(data_loader, metadata=metadata)

        preprocessed_data = preprocessed_loader.load_all()

        attached_columns = None
        if set(preprocessed_data.columns) != set(self.multi_x_join.join_tables):
            warnings.warn("Detected columns deleting by data processors, we will attach original data to train RFECV.")
            missing_columns = set(self.multi_x_join.join_tables) - set(preprocessed_data.columns)
            attached_columns = pd.DataFrame(self.multi_x_join.join_tables[list(missing_columns)])

        self.to_training = MultiTableRfecvCTGAN._argument_tables(
            join_tables=preprocessed_data,  # TODO 优化
            attached_columns=attached_columns,
            tables_name=self.multi_x_join.origin_tables.keys(),
            columns_table_map=self.multi_x_join.join_columns_table_name_map,
            join_columns_map=self.multi_x_join.join_columns_map,
            metadata=metadata)

    def _data_preprocess(self, dataloader, metadata):
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
    def _do_rfecv(target_column_name: str, data: pd.DataFrame, metadata) -> set[str]:
        def _is_categorical(column: pd.Series | pd.DataFrame) -> bool:
            return int(column.nunique()) <= CATEGORY_THRESHOLD

        #     return column.dtype == "object" or column.nunique() <= CATEGORY_THRESHOLD

        # data = data.copy()
        # TODO 这是临时使用的onehot，每次会重复很多次。考虑优化
        discrete_columns = set(metadata.get("discrete_columns"))
        onehot_columns_map = {}
        for col in discrete_columns:
            if col not in data:
                continue
            if target_column_name != col and _is_categorical(data[col]):
                # Onehot Encode
                onehot_data = pd.get_dummies(data[col], prefix=col, prefix_sep="_Onehot_")
                data = data.drop(col, axis=1)
                data = pd.concat([data, onehot_data], axis=1)
                onehot_columns_map.update({
                    c: col for c in onehot_data.columns
                })
            else:
                # Label encode
                mapper = {v: k for k, v in enumerate(sorted(pd.unique(data[col]), key=lambda x: x))}
                data[col] = data[col].map(mapper)

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
                         attached_columns: pd.DataFrame = None,
                         tables_name: List[str],
                         columns_table_map: Dict[str, List[str]],
                         join_columns_map: Dict[str, Dict[str, str]],
                         metadata: Metadata) -> List:
        convert_columns_map = {
            key: {v: k for k, v in value.items()} for key, value in join_columns_map.items()
        }

        if attached_columns is None:
            attached_columns = pd.DataFrame()
        for tbn in convert_columns_map:
            assert set(convert_columns_map[tbn].values()).issubset(
                set(join_tables.columns).union(attached_columns.columns))
        attached_tables = pd.concat([attached_columns, join_tables], axis=1)

        selected_columns = set()
        to_training = []

        # testset = frozenset(join_tables.columns)

        # 对每一个表的每一列 和 未被选择的列 进行 RFECV
        for table_name in tables_name:

            # assert testset == frozenset(join_tables.columns)
            # 其他表的非选择的列
            other_table_names = set(tables_name) - {table_name}
            all_columns_name = set(join_tables.columns) - selected_columns
            other_table_columns_name = set(
                cn for cn in all_columns_name
                if other_table_names.intersection(set(columns_table_map[cn])))

            current_table_columns = convert_columns_map[table_name].values()

            # assert not (set(current_table_columns) - set(join_tables.columns))
            # assert set(current_table_columns).issubset(set(join_tables.columns))

            if not set(current_table_columns) - selected_columns:
                to_training.append(RFECV_CTGANTrainingData(
                    table_name=table_name,
                ))
                continue
            # 对每个表的【每一列】
            target_column_name_selected_map = {}
            to_training_columns = set(current_table_columns)
            for target_column_name in current_table_columns:
                # 如果是参考列，只算全大表第一次的RFECV，后续选择不再考虑，但还要跟随本表其他列进行计算RFECV。
                # 如果是独特列，同理。
                # if MultiTableCTGAN.is_ref_column(target_column_name):
                if target_column_name in selected_columns:
                    continue
                to_rfe_columns_name = set(current_table_columns).union(other_table_columns_name)
                try:
                    to_rfe_columns = join_tables[list(to_rfe_columns_name)].copy()
                except KeyError:
                    to_rfe_columns = attached_tables[list(to_rfe_columns_name)].copy()

                # rfe TODO
                current_selected_columns = set(
                    MultiTableRfecvCTGAN._do_rfecv(target_column_name, to_rfe_columns, metadata)
                )

                # current_selected_columns = random.choices(list(to_rfe_columns_name), k=3)

                selected_columns = selected_columns.union(current_selected_columns)
                selected_columns.add(target_column_name)
                to_training_columns = to_training_columns.union(current_selected_columns)
                target_column_name_selected_map.update({
                    target_column_name: current_selected_columns
                })
            # 每个表生成一个模型
            current_attached_columns = set(attached_columns.columns).intersection(to_training_columns)
            to_training_tables = join_tables[list(to_training_columns - current_attached_columns)].copy()
            to_training.append(RFECV_CTGANTrainingData(**{
                'table_name': table_name,
                'attached_columns': current_attached_columns,
                'target_column_name_selected_map': target_column_name_selected_map,
                'tables': to_training_tables,
                'columns': to_training_columns  # 后面的列只从这里取出来，后面生成的不要
            }))
        return to_training
