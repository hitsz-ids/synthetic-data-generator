import pytest

from sdgx.models.manager import ModelManager
from sdgx.utils.io.csv_utils import get_demo_single_table


@pytest.fixture
def model_manager():
    yield ModelManager()


def test_model_manager(model_manager: ModelManager):
    assert "ctgan" in model_manager.registed_model

    model = model_manager.init_model("ctgan", epochs=1)
    demo_data, discrete_cols = get_demo_single_table()
    model.fit(demo_data, discrete_cols)

    # 生成合成数据
    sampled_data = model.sample(1000)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
