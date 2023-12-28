from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel

from sdgx.data_loader import DataLoader
from sdgx.data_models.inspectors.manager import InspectorManager
from sdgx.exceptions import MetadataInitError, MetadataInvalidError
from sdgx.utils import logger


class Metadata(BaseModel):
    """Metadata

    This metadata is mainly used to describe the data types of all columns in a single data table.

    For each column, there should be an instance of the Data Type object.

    Args:
        primary_keys(List[str]): The primary key, a field used to uniquely identify each row in the table.
        The primary key of each row must be unique and not empty.

        column_list(list[str]): list of the comlumn name in the table, other columns lists are used to store column information.
    """

    primary_keys: List[str] = []
    """
    primary_keys is used to store single primary key or composite primary key
    """

    column_list: List[str] = []
    """"
    column_list is used to store all columns' name
    """

    # other columns lists are used to store column information
    # here are 5 basic data types
    id_columns: List[str] = []
    numeric_columns: List[str] = []
    bool_columns: List[str] = []
    discrete_columns: List[str] = []
    datetime_columns: List[str] = []

    # version info
    metadata_version: str = "1.0"
    _extend: Dict[str, Any] = {}
    """
    For extend information, use ``get`` and ``set``
    """

    def get(self, key: str, default=None) -> Any:
        return getattr(self, key, getattr(self._extend, key, default))

    def set(self, key: str, value: Any):
        if key == "_extend":
            raise MetadataInitError("Cannot set _extend directly")

        if key in self.model_fields:
            setattr(self, key, value)
        else:
            self._extend[key] = value

    def update(self, attributes: dict[str, Any]):
        for k, v in attributes.items():
            self.set(k, v)

        return self

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        max_chunk: int = 10,
        primary_keys: List[str] = None,
        include_inspectors: list[str] | None = None,
        exclude_inspectors: list[str] | None = None,
        inspector_init_kwargs: dict[str, Any] | None = None,
    ) -> "Metadata":
        """Initialize a metadata from DataLoader and Inspectors

        Args:
            dataloader(DataLoader): the input DataLoader.

            max_chunk(int): max chunk count.

            primary_keys(list[str]): primary keys, see :class:`~sdgx.data_models.metadata.Metadata` for more details.

            include_inspectors(list[str]): data type inspectors that should included in this metadata (table).

            exclude_inspectors(list[str]): data type inspectors that should NOT included in this metadata (table).

            inspector_init_kwargs(dict): inspector args.
        """
        logger.info("Inspecting metadata...")
        inspectors = InspectorManager().init_inspcetors(
            include_inspectors, exclude_inspectors, **(inspector_init_kwargs or {})
        )
        for i, chunk in enumerate(dataloader.iter()):
            for inspector in inspectors:
                inspector.fit(chunk)
            if all(i.ready for i in inspectors) or i > max_chunk:
                break

        # If primary_key is not specified, use the first column (in list).
        if primary_keys is None:
            primary_keys = [dataloader.columns()[0]]

        metadata = Metadata(primary_keys=primary_keys, column_list=dataloader.columns())
        for inspector in inspectors:
            metadata.update(inspector.inspect())

        return metadata

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        include_inspectors: list[str] | None = None,
        exclude_inspectors: list[str] | None = None,
        inspector_init_kwargs: dict[str, Any] | None = None,
    ) -> "Metadata":
        inspectors = InspectorManager().init_inspcetors(
            include_inspectors, exclude_inspectors, **(inspector_init_kwargs or {})
        )
        for inspector in inspectors:
            inspector.fit(df)

        metadata = Metadata(primary_keys=[df.columns[0]], column_list=list(df.columns))
        for inspector in inspectors:
            metadata.update(inspector.inspect())

        return metadata

    def save(self, path: str | Path):
        with path.open("w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def load(cls, path: str | Path) -> "Metadata":
        path = Path(path).expanduser().resolve()
        attributes = json.load(path.open("r"))
        return Metadata().update(attributes)

    def check_single_primary_key(self, input_key: str):
        """Check whether a primary key in column_list and has ID data type.

        Args:
            input_key(str): the input primary_key str
        """

        if input_key not in self.column_list:
            raise MetadataInvalidError(f"Primary Key {input_key} not Exist in columns.")
        if input_key not in self.id_columns:
            raise MetadataInvalidError(f"Primary Key {input_key} should has ID DataType.")

    def get_all_data_type_columns(self):
        """Get all column names from `self.xxx_columns`.

        All Lists with the suffix _columns in model fields and extend fields need to be collected.
        All defined column names will be counted.

        Returns:
            all_dtype_cols(set): set of all column names.
        """
        all_dtype_cols = set()

        # search the model fields and extend fields
        for each_key in list(self.model_fields.keys()) + list(self._extend.keys()):
            if each_key.endswith("_columns"):
                column_names = self.get(each_key)
                all_dtype_cols = all_dtype_cols.union(set(column_names))

        return all_dtype_cols

    def check(self):
        """Checks column info.

        When passing as input to the next module, perform necessary checks, including:
            -Is the primary key correctly defined(in column list) and has ID data type.
            -Is there any missing definition of each column in table.
            -Are there any unknown columns that have been incorrectly updated.
        """
        # check primary key in column_list and has ID data type
        for each_key in self.primary_keys:
            self.check_single_primary_key(each_key)

        all_dtype_columns = self.get_all_data_type_columns()

        # check missing columns
        if set(self.column_list) - set(all_dtype_columns):
            raise MetadataInvalidError(
                f"Undefined data type for column {set(self.column_list) - set(all_dtype_columns)}."
            )

        # check unfamiliar columns in dtypes
        if set(all_dtype_columns) - set(self.column_list):
            raise MetadataInvalidError(
                f"Found undefined column: {set(all_dtype_columns) - set(self.column_list)}."
            )

        logger.debug("Metadata check succeed.")

    def update_primary_key(self, primary_keys: List[str]):
        """Update the primary key of the table

        When update the primary key, the original primary key will be erased.

        Args:
            primary_keys(List[str]): the primary keys of this table.
        """

        if not isinstance(primary_keys, List):
            raise MetadataInvalidError("Primary key should be a list.")

        for each_key in primary_keys:
            if each_key not in self.column_list:
                raise MetadataInvalidError("Primary key not exist in table columns.")

        self.primary_keys = primary_keys

        logger.info(f"Primary Key updated: {primary_keys}.")
