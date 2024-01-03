from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd
from pydantic import BaseModel

from sdgx.data_loader import DataLoader
from sdgx.data_models.inspectors.manager import InspectorManager
from sdgx.exceptions import MetadataInitError, MetadataInvalidError
from sdgx.utils import logger


class Metadata(BaseModel):
    """
    Metadata is mainly used to describe the data types of all columns in a single data table.

    For each column, there should be an instance of the Data Type object.

    .. Note::

        Use ``get``, ``set``, ``add``, ``delete`` to update the metadata.

    Args:
        primary_keys(List[str]): The primary key, a field used to uniquely identify each row in the table.
        The primary key of each row must be unique and not empty.

        column_list(list[str]): list of the comlumn name in the table, other columns lists are used to store column information.
    """

    primary_keys: Set[str] = set()
    """
    primary_keys is used to store single primary key or composite primary key
    """

    column_list: Set[str] = set()
    """"
    column_list is used to store all columns' name
    """

    # other columns lists are used to store column information
    # here are 5 basic data types
    id_columns: Set[str] = set()
    numeric_columns: Set[str] = set()
    bool_columns: Set[str] = set()
    discrete_columns: Set[str] = set()
    datetime_columns: Set[str] = set()

    # version info
    version: str = "1.0"
    _extend: Dict[str, Set[str]] = defaultdict(set)
    """
    For extend information, use ``get`` and ``set``
    """

    def __eq__(self, other):
        if not isinstance(other, Metadata):
            return super().__eq__(other)
        return (
            all(self.get(key) == other.get(key) for key in self.get_all_data_type_columns())
            and self.version == other.version
        )

    def get(self, key: str) -> Set[str]:
        return getattr(self, key, self._extend[key])

    def set(self, key: str, value: Any):
        if key == "_extend":
            raise MetadataInitError("Cannot set _extend directly")

        old_value = self.get(key)

        if isinstance(old_value, Iterable) and not isinstance(old_value, str):
            # list, set, tuple...
            value = value if isinstance(value, Iterable) and not isinstance(value, str) else [value]
            value = type(old_value)(value)

        if key in self.model_fields:
            setattr(self, key, value)
        else:
            self._extend[key] = value

    def add(self, key: str, values: str | Iterable[str]):
        values = (
            values if isinstance(values, Iterable) and not isinstance(values, str) else [values]
        )

        for value in values:
            self.get(key).add(value)

    def delete(self, key: str, value: str):
        try:
            self.get(key).remove(value)
        except KeyError:
            pass

    def update(self, attributes: dict[str, Any]):
        for k, v in attributes.items():
            self.add(k, v)

        return self

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        max_chunk: int = 10,
        primary_keys: set[str] = None,
        include_inspectors: Iterable[str] | None = None,
        exclude_inspectors: Iterable[str] | None = None,
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

        if primary_keys is None:
            primary_keys = set()

        metadata = Metadata(primary_keys=primary_keys, column_list=set(dataloader.columns()))
        for inspector in inspectors:
            metadata.update(inspector.inspect())
        if not primary_keys:
            metadata.update_primary_key(metadata.id_columns)

        metadata.check()
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

        metadata = Metadata(primary_keys=[df.columns[0]], column_list=set(df.columns))
        for inspector in inspectors:
            metadata.update(inspector.inspect())
        metadata.check()
        return metadata

    def _dump_json(self):
        return self.model_dump_json()

    def save(self, path: str | Path):
        with path.open("w") as f:
            f.write(self._dump_json())

    @classmethod
    def load(cls, path: str | Path) -> "Metadata":
        path = Path(path).expanduser().resolve()
        attributes = json.load(path.open("r"))
        version = attributes.get("version", None)
        if version:
            cls.upgrade(version, attributes)

        m = Metadata()
        for k, v in attributes.items():
            m.set(k, v)
        return m

    @classmethod
    def upgrade(cls, old_version: str, fields: dict[str, Any]) -> None:
        pass

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

    def update_primary_key(self, primary_keys: Iterable[str] | str):
        """Update the primary key of the table

        When update the primary key, the original primary key will be erased.

        Args:
            primary_keys(Iterable[str]): the primary keys of this table.
        """

        if not isinstance(primary_keys, Iterable) and not isinstance(primary_keys, str):
            raise MetadataInvalidError("Primary key should be Iterable or str.")
        primary_keys = set(primary_keys if isinstance(primary_keys, Iterable) else [primary_keys])

        if not primary_keys.issubset(self.column_list):
            raise MetadataInvalidError("Primary key not exist in table columns.")

        self.primary_keys = primary_keys

        logger.info(f"Primary Key updated: {primary_keys}.")
