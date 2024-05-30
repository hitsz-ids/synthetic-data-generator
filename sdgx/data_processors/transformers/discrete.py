from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.models.components.optimize.ndarray_loader import NDArrayLoader
from sdgx.utils import logger


class DiscreteTransformer(Transformer):
    """
    DiscreteTransformer is an important component of sdgx, used to handle discrete columns.

    By default, DiscreteTransformer will perform one-hot encoding of discrete columns, and issue a warning message when dimensionality explosion occurs.
    """

    discrete_columns: list = []
    """
    Record which columns are of discrete type.
    """

    one_hot_warning_cnt = 512

    one_hot_encoders: dict = {}

    one_hot_column_names: dict = {}

    onehot_encoder_handle_unknown: str = "ignore"

    def fit(self, metadata: Metadata, tabular_data: DataLoader | pd.DataFrame):
        """
        Fit method for the DiscreteTransformer.
        """

        logger.info("Fitting using DiscreteTransformer...")

        self.discrete_columns = metadata.get("discrete_columns")

        # remove datetime columns from discrete columns
        # because datetime columns are converted into timestamps
        datetime_columns = metadata.get("datetime_columns")

        # no discrete columns
        if len(self.discrete_columns) == 0:
            logger.info("Fitting using DiscreteTransformer... Finished (No Columns).")
            return

        # fit each columns
        for each_datgetime_col in datetime_columns:
            if each_datgetime_col in self.discrete_columns:
                self.discrete_columns.remove(each_datgetime_col)
                logger.info(f"Datetime column {each_datgetime_col} removed from discrete column.")

        # then, there are >= 1 discrete colums
        for each_col in self.discrete_columns:
            # fit each column
            self._fit_column(each_col, tabular_data[[each_col]])

        logger.info("Fitting using DiscreteTransformer... Finished.")
        self.fitted = True

        return

    def _fit_column(self, column_name: str, column_data: pd.DataFrame):
        """
        Fit every discrete column in `_fit_column`.

        Args:
            - column_data (pd.DataFrame): A dataframe containing a column.
            - column_name: str: column name.
        """

        self.one_hot_encoders[column_name] = OneHotEncoder(
            handle_unknown=self.onehot_encoder_handle_unknown, sparse_output=False
        )
        # fit the column data
        self.one_hot_encoders[column_name].fit(column_data)

        logger.debug(f"Discrete column {column_name} fitted.")

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle discrete values in the input data.
        """

        logger.info("Converting data using DiscreteTransformer...")

        # transform every discrete column into one-hot encoded columns
        if len(self.discrete_columns) == 0:
            logger.info("Converting data using DiscreteTransformer... Finished (No column).")
            return

        processed_data = raw_data.copy()

        for each_col in self.discrete_columns:
            # 1- transform each column
            new_onehot_columns = self.one_hot_encoders[each_col].transform(raw_data[[each_col]])
            new_onehot_column_names = self.one_hot_encoders[each_col].get_feature_names_out()
            self.one_hot_column_names[each_col] = new_onehot_column_names

            # logger warning if too many columns
            if len(new_onehot_column_names) > self.one_hot_warning_cnt:
                logger.warning(
                    f"Column {each_col} has too many discrete values ({len(new_onehot_column_names)} values), may consider as a continous column?"
                )

            # 2- add new_onehot_column_set into the original dataframe, record the column name ?
            processed_data = self.attach_columns(
                processed_data, pd.DataFrame(new_onehot_columns, columns=new_onehot_column_names)
            )

            logger.debug(f"Column {each_col} converted.")

        logger.info(f"Processed data shape: {processed_data.shape}.")

        logger.info("Converting data using DiscreteTransformer... Finished.")

        processed_data = self.remove_columns(processed_data, self.discrete_columns)

        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the transformer.

        Args:
            - processed_data (pd.DataFrame): A dataframe containing onehot encoded columns.

        Returns:
            - pd.DataFrame: inverse transformed processed data.
        """

        reversed_data = processed_data.copy()

        # for each discrete col
        for each_col in self.discrete_columns:
            # 1- get one-hot column sets from processed data
            one_hot_column_set = processed_data[self.one_hot_column_names[each_col]]
            # 2- inverse convert using ohe
            res_column_data = self.one_hot_encoders[each_col].inverse_transform(
                pd.DataFrame(one_hot_column_set, columns=self.one_hot_column_names[each_col])
            )
            # 3- put original column back to reversed_data
            reversed_data = self.attach_columns(
                reversed_data, pd.DataFrame(res_column_data, columns=[each_col])
            )
            reversed_data = self.remove_columns(reversed_data, self.one_hot_column_names[each_col])

        logger.info("Data inverse-converted by DiscreteTransformer.")

        return reversed_data

    pass


@hookimpl
def register(manager):
    manager.register("DiscreteTransformer", DiscreteTransformer)
