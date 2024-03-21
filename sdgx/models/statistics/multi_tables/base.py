from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import warnings

from sdgx.data_models.relationship import Relationship

from sdgx.models.statistics.single_table.copula import GaussianCopulaSynthesizer

from sdgx.data_models.metadata import Metadata

from sdgx.data_loader import DataLoader
from sdgx.data_models.combiner import MetadataCombiner
from sdgx.exceptions import SynthesizerInitError
from sdgx.log import logger
from sdgx.models.base import SynthesizerModel


class MultiTableSynthesizerModel(SynthesizerModel):
    """MultiTableSynthesizerModel

    The base model of multi-table statistic models.
    """

    metadata_combiner: MetadataCombiner = None
    """
    metadata_combiner is a sdgx builtin class, it stores all tables' metadata and relationships.

    This parameter must be specified when initializing the multi-table class.
    """

    tables_data_frame: Dict[str, Any] = defaultdict()
    """
    tables_data_frame is a dict contains every table's csv data frame.
    For a small amount of data, this scheme can be used.
    """

    tables_data_loader: Dict[str, Any] = defaultdict()
    """
    tables_data_loader is a dict contains every table's data loader.
    """

    _parent_id: List = []
    """
    _parent_id is used to store all parent table's parimary keys in list.
    """

    _table_synthesizers: Dict[str, Any] = {}
    """
    _table_synthesizers is a dict to store model for each table.
    """

    parent_map: Dict = defaultdict(set)
    """
    The mapping from all child tables to their parent table.
    """

    child_map: Dict = defaultdict(set)
    """
    The mapping from all parent tabels to their child table.
    """
    DEFAULT_SYNTHESIZER_KWARGS = None
    _synthesizer = GaussianCopulaSynthesizer

    def _initialize_models(self):
        for table_name, table_metadata in self.tables.items():
            synthesizer_parameters = self._table_parameters.get(table_name, {})
            self._table_synthesizers[table_name] = self._synthesizer(
                metadata=table_metadata,
                locales=self.locales,
                **synthesizer_parameters
            )
            if self.DEFAULT_SYNTHESIZER_KWARGS:
                self._table_parameters[table_name] = deepcopy(self.DEFAULT_SYNTHESIZER_KWARGS)

    def __init__(self, metadata_combiner: MetadataCombiner, locales=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #tables存储单表的metadata
        self.tables = {}

        self.locales = locales
        self.verbose = False

        self.extended_columns = defaultdict(dict)
        self._table_synthesizers = {}
        self._table_parameters = defaultdict(dict)

        self._initialize_models()
        self._fitted = False
        self._fitted_date = None

        self.metadata_combiner = metadata_combiner
        self._calculate_parent_and_child_map()
        self.check()

    # ---------------------------------------------------------------------------
    #这一部分作为多表操作或许应该放在combiner上
    def remove_primary_key(self, table_name):
        """Remove the primary key from the given table.

        Removes the primary key from the given table. Also removes any relationships that
        reference that table's primary key, including all relationships in which the given
        table is a parent table.

        Args:
            table_name (str):
                The name of the table to remove the primary key from.
        """
        primary_key = self.tables[table_name].primary_key
        self.tables[table_name].remove_primary_key()

        for relationship in self.metadata_combiner.relationships:
            parent_table = relationship.parent_table
            child_table = relationship.child_table
            foreign_key = self._get_foreign_keys(parent_table, child_table)
            if ((child_table == table_name and foreign_key == primary_key) or
                    parent_table == table_name):
                self.metadata_combiner.relationships.remove(relationship)

    def add_column(self, table_name, column_name, **kwargs):
        """Add a column to a table in the ``MultiTableMetadata``.

        Args:
            table_name (str):
                Name of the table to add the column to.
            column_name (str):
                The column name to be added.
            **kwargs (type):
                Any additional key word arguments for the column, where ``sdtype`` is required.

        Raises:
            - ``InvalidMetadataError`` if the column already exists.
            - ``InvalidMetadataError`` if the ``kwargs`` do not contain ``sdtype``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              given ``sdtype``.
            - ``InvalidMetadataError`` if the table doesn't exist in the ``MultiTableMetadata``.
        """
        table = self.tables.get(table_name)
        table.add_column(column_name, **kwargs)

    def update_column(self, table_name, column_name, **kwargs):
        """Update an existing column for a table in the ``MultiTableMetadata``.

        Args:
            table_name (str):
                Name of table the column belongs to.
            column_name (str):
                The column name to be updated.
            **kwargs (type):
                Any key word arguments that describe metadata for the column.

        Raises:
            - ``InvalidMetadataError`` if the column doesn't already exist in the
              ``SingleTableMetadata``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              current ``sdtype``.
            - ``InvalidMetadataError`` if the table doesn't exist in the ``MultiTableMetadata``.
        """
        table = self.tables.get(table_name)
        table.update_column(column_name, **kwargs)

    def set_primary_key(self, table_name, column_name):
        """Set the primary key of a table.

        Args:
            table_name (str):
                Name of the table to set the primary key.
            column_name (str, tulple[str]):
                Name (or tuple of names) of the primary key column(s).
        """
        self.tables[table_name].set_primary_key(column_name)

    def add_alternate_keys(self, table_name, column_names):
        """Set the alternate keys of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence key.
            column_names (list[str], list[tuple]):
                List of names (or tuple of names) of the alternate key columns.
        """
        self.tables[table_name].add_alternate_keys(column_names)

    def set_sequence_index(self, table_name, column_name):
        """Set the sequence index of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence index.
            column_name (str):
                Name of the sequence index column.
        """
        warnings.warn('Sequential modeling is not yet supported on SDV Multi Table models.')
        self.tables[table_name].set_sequence_index(column_name)

    def add_column_relationship(self, table_name, relationship_type, column_names):
        """Add a column relationship to a table in the metadata.

        Args:
            table_name (str):
                The name of the table to add this relationship to.
            relationship_type (str):
                The type of the relationship.
            column_names (list[str]):
                The list of column names involved in this relationship.
        """
        foreign_keys = self._get_all_foreign_keys(table_name)
        relationships = [{'type': relationship_type, 'column_names': column_names}] + \
                        self.tables[table_name].column_relationships
        self.tables[table_name].add_column_relationship(relationship_type, column_names)

    def add_table(self, table_name):
        """Add a table to the metadata.

        Args:
            table_name (str):
                The name of the table to add to the metadata.

        Raises:
            Raises ``InvalidMetadataError`` if ``table_name`` is not valid.
        """
        if not isinstance(table_name, str) or table_name == '':
            raise Exception(
                "Invalid table name (''). The table name must be a non-empty string."
            )

        if table_name in self.tables:
            raise Exception(
                f"Cannot add a table named '{table_name}' because it already exists in the "
                'metadata. Please choose a different name.'
            )

        self.tables[table_name] = Metadata()

    def _set_metadata_dict(self, metadata):
        """Set a ``metadata`` dictionary to the current instance.

        Args:
            metadata (dict):
                Python dictionary representing a ``MultiTableMetadata`` object.
        """
        for table_name, table_dict in metadata.get('tables', {}).items():
            self.tables[table_name] = Metadata.load_from_dict(table_dict)

        for relationship in metadata.get('relationships', []):
            parent_table = relationship['parent_table_name']
            child_table = relationship['child_table_name']
            foreign_keys = relationship['foreign_keys']
            self.metadata_combiner.relationships.append(
                Relationship.build(
                    parent_table=parent_table,
                    child_table=child_table,
                    foreign_keys=foreign_keys
                )
            )

    def load_from_dict(self, metadata_dict):
        """Create a ``MultiTableMetadata`` instance from a python ``dict``.

        Args:
            metadata_dict (dict):
                Python dictionary representing a ``MultiTableMetadata`` object.

        Returns:
            Instance of ``MultiTableMetadata``.
        """
        self._set_metadata_dict(metadata_dict)
        self._calculate_parent_and_child_map()
        self._initialize_models()

    # ---------------------------------------------------------------------------
    def _calculate_parent_and_child_map(self):
        """Get the mapping from all parent tables to self._parent_map
        - key(str) is a child map;
        - value(str) is the parent map.
        """
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            parent_table = each_relationship.parent_table
            child_table = each_relationship.child_table
            self.parent_map[child_table].add(parent_table)
            self.child_map[parent_table].add(child_table)

    def _get_foreign_keys(self, parent_table, child_table):
        """Get the foreign key list from a relationship"""

        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            # find the exact relationship and return foreign keys
            if (
                    each_relationship.parent_table == parent_table
                    and each_relationship.child_table == child_table
            ):
                return [each_relationship.foreign_keys[0][1]]
        return []

    def _get_all_foreign_keys(self, child_table):
        """Given a child table, return ALL foreign keys from metadata."""
        all_foreign_keys = []
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            # find the exact relationship and return foreign keys
            if each_relationship.child_table == child_table:
                foreign_key = each_relationship.foreign_keys[0][1]
                all_foreign_keys.append(foreign_key)

        return all_foreign_keys

    def _finalize(self, sampled_data):
        """Finalize the"""
        raise NotImplementedError

    def check(self, check_circular=True):
        """Excute necessary checks

        - check access type
        - check metadata_combiner
        - check relationship
        - check each metadata
        - validate circular relationships
        - validate child map_circular relationship
        - validate all tables connect relationship
        - validate column relationships foreign keys
        """
        # self._check_access_type()

        if not isinstance(self.metadata_combiner, MetadataCombiner):
            raise SynthesizerInitError("Wrong Metadata Combiner found.")
        pass

    def fit(
            self, dataloader: Dict[str, DataLoader], raw_data: Dict[str, pd.DataFrame], *args, **kwargs
    ):
        """
        Fit the model using the given metadata and dataloader.

        Args:
            dataloader (Dict[str, DataLoader]): The dataloader to use to fit the model.
            raw_data (Dict[str, pd.DataFrame]): The raw pd.DataFrame to use to fit the model.
        """
        """Fit this model to the original data.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame`` in the raw format
                (before any transformations).
        """
        self._fitted = False
        #由于preprocess问题，想要用fit可以直接使用model_tables函数
        processed_data = self.preprocess(raw_data)
        self.fit_processed_data(processed_data)

    def sample(self, count: float, *args, **kwargs) -> pd.DataFrame:
        """
        Sample data from the model.

        Args:
            count (float): The number of samples to generate.
                            这里是浮点数，乘以原表列数等于sample的列数

        Returns:
            pd.DataFrame: The generated data.
        """

        raise NotImplementedError

    def save(self, save_dir: str | Path):
        pass

    # @classmethod
    # def load(target_path: str | Path):
    #     pass

    # -------------------------------------------------------------------------------
    def _get_root_parents(self):
        """Get the set of root parents in the graph."""
        non_root_tables = set(self.parent_map.keys())
        root_parents = set(self.tables.keys()) - non_root_tables
        return root_parents

    def set_address_columns(self, table_name, column_names, anonymization_level='full'):
        """Set the address multi-column transformer.

        Args:
            table_name (str):
                The name of the table for which the address transformer should be set.
            column_names (tuple[str]):
                The column names to be used for the address transformer.
            anonymization_level (str):
                The anonymization level to use for the address transformer.
        """
        self._table_synthesizers[table_name].set_address_columns(column_names, anonymization_level)

    def get_table_parameters(self, table_name):
        """Return the parameters for the given table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary representing the parameters that will be used to instantiate the
                table's synthesizer.
        """
        table_synthesizer = self._table_synthesizers.get(table_name)
        if not table_synthesizer:
            table_params = {'table_synthesizer': None, 'table_parameters': {}}
        else:
            table_params = {
                'table_synthesizer': type(table_synthesizer).__name__,
                'table_parameters': table_synthesizer.get_parameters()
            }

        return table_params

    def set_table_parameters(self, table_name, table_parameters):
        """Update the table's synthesizer instantiation parameters.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.
            table_parameters (dict):
                A dictionary with the parameters as keys and the values to be used to instantiate
                the table's synthesizer.
        """
        self._table_synthesizers[table_name] = self._synthesizer(
            metadata=self.tables[table_name],
            **table_parameters
        )
        self._table_parameters[table_name].update(deepcopy(table_parameters))

    def _assign_table_transformers(self, synthesizer, table_name, table_data):
        """Update the ``synthesizer`` to ignore the foreign keys while preprocessing the data."""
        synthesizer.auto_assign_transformers(table_data)
        foreign_key_columns = self._get_all_foreign_keys(table_name)
        column_name_to_transformers = {
            column_name: None
            for column_name in foreign_key_columns
        }
        synthesizer.update_transformers(column_name_to_transformers)

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (dict):
                Mapping of table name to pandas.DataFrame.

        Raises:
            InvalidDataError:
                If a table of the data is not present in the metadata.
        """
        for table_name, table_data in data.items():
            synthesizer = self._table_synthesizers[table_name]
            self._assign_table_transformers(synthesizer, table_name, table_data)

    def get_transformers(self, table_name):
        """Get a dictionary mapping of ``column_name`` and ``rdt.transformers``.

        A dictionary representing the column names and the transformers that will be used
        to transform the data.

        Args:
            table_name (string):
                The name of the table of which to get the transformers.

        Returns:
            dict:
                A dictionary mapping with column names and transformers.

        Raises:
            ValueError:
                If ``table_name`` is not present in the metadata.
        """
        return self._table_synthesizers[table_name]._transformer

    def preprocess(self, data):
        """Transform the raw data to numerical space.
        这里就需要用copula的preprocess了，需要dataprocesser
        后面或许解耦后能直接把synthessizer对应字段改成dataprocesser？
        目前暂时不能用
        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame``.

        Returns:
            dict:
                A dictionary with the preprocessed data.
        """
        if self._fitted:
            warnings.warn(
                'This model has already been fitted. To use the new preprocessed data, '
                "please refit the model using 'fit' or 'fit_processed_data'."
            )

        processed_data = {}
        for table_name, table_data in data.items():
            synthesizer = self._table_synthesizers[table_name]
            self._assign_table_transformers(synthesizer, table_name, table_data)
            processed_data[table_name] = synthesizer._preprocess(table_data)

        return processed_data

    def model_tables(self, augmented_data):
        """Model the augmented tables.

        Args:
            augmented_data (dict):
                Dictionary mapping each table name to an augmented ``pandas.DataFrame``.
        """
        raise NotImplementedError()

    def get_extended_tables(self, processed_data):
        """Augment the processed data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        raise NotImplementedError()

    def fit_processed_data(self, processed_data):
        """Fit this model to the transformed data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        augmented_data = self.get_extended_tables(processed_data)
        self.model_tables(augmented_data)
        self._fitted = True

# -------------------------------------------------------------------------------
