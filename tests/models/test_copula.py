from sdgx.models.statistics.single_table.copula import GaussianCopulaSynthesizer
from sdgx.utils import get_demo_single_table


def test_gaussian_copula(demo_single_table_path):
    demo_data, discrete_cols = get_demo_single_table(demo_single_table_path.parent)
    model = GaussianCopulaSynthesizer(discrete_cols)
    model.fit(demo_data)

    sampled_data = model.sample(10)
    assert len(sampled_data) == 10
