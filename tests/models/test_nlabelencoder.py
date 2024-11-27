import shutil

import pandas as pd
import pytest

from sdgx.models.components.sdv_rdt.transformers.categorical import NormalizedLabelEncoder

@pytest.fixture(scope='module')
def data_test():
    return pd.DataFrame({
        'x': [str(i) for i in range(100)],
        'y': [str(-i) for i in range(50)]*2,
        'z': [str(i) for i in range(25)]*4
    }, columns=['x', 'y', 'z'])

def test_encoder(
    data_test: pd.DataFrame
):

    for col in ['x', 'y', 'z']:
        nlabel_encoder = NormalizedLabelEncoder()
        nlabel_encoder.fit(data_test, col)
        td = nlabel_encoder.transform(data_test.copy())
        rd = nlabel_encoder.reverse_transform(td.copy())
        td.rename(columns={f'{col}.value': f'{col}'}, inplace=True)
        assert (rd[col].sort_values().values == data_test[col].sort_values().values).any()
        assert (td[col] >= 0).any()
        assert (td[col] <= 1).any()
        assert td[col].shape == data_test[col].shape
        assert len(td[col].unique()) == len(data_test[col].unique())

if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
