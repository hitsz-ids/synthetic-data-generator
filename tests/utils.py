"""Utils for testing."""
import pandas as pd
from sdgx.data_models.relationship import KeyTuple


class DataFrameMatcher:
    """Match a given Pandas DataFrame in a mock function call."""

    def __init__(self, df):
        self.df = df

    def __eq__(self, other):
        pd.testing.assert_frame_equal(self.df, other)
        return True


class SeriesMatcher:
    """Match a given Pandas Series in a mock function call."""

    def __init__(self, series):
        self.series = series

    def __eq__(self, other):
        pd.testing.assert_series_equal(self.series, other)
        return True


def get_multi_table_metadata():
    """Return a ``MultiTableMetadata`` object to be used with tests."""
    dict_metadata = {
        'tables': {
            'nesreca': {
                'primary_key': 'id_nesreca',
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'nesreca_val': {'sdtype': 'numerical'}
                }
            },
            'oseba': {
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'oseba_val': {'sdtype': 'numerical'}
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {
                    'id_upravna_enota': {'sdtype': 'id'},
                    'upravna_val': {'sdtype': 'numerical'}
                }
            }
        },
        'relationships': [
            {
                'parent_table_name': 'upravna_enota',
                'child_table_name': 'nesreca',
                'foreign_keys': [KeyTuple("id_upravna_enota", "upravna_enota")]
            },
            {
                'parent_table_name': 'upravna_enota',
                'child_table_name': 'oseba',
                'foreign_keys': [KeyTuple("id_upravna_enota", "upravna_enota")]
            },
            {
                'parent_table_name': 'nesreca',
                'child_table_name': 'oseba',
                'foreign_keys': [KeyTuple("id_nesreca", "id_nesreca")]
            }
        ]
    }

    return dict_metadata


def get_multi_table_data():
    """Return a dictionary containing some data for multi table."""
    data = {
        'nesreca': pd.DataFrame({
            'id_nesreca': list(range(4)),
            'upravna_enota': list(range(4)),
            'nesreca_val': list(range(4))
        }),
        'oseba': pd.DataFrame({
            'upravna_enota': list(range(4)),
            'id_nesreca': list(range(4)),
            'oseba_val': list(range(4))
        }),
        'upravna_enota': pd.DataFrame({
            'id_upravna_enota': list(range(4)),
            'upravna_val': list(range(4))
        }),
    }

    return data
