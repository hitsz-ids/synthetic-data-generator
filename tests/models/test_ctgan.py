import shutil

import pytest

from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel


@pytest.fixture
def ctgan():
    yield CTGANSynthesizerModel(epochs=1)


@pytest.fixture
def save_model_dir(tmp_path):
    dirname = tmp_path / "model"
    yield dirname
    shutil.rmtree(dirname, ignore_errors=True)


def assert_sampled_data(dummy_single_table_data_loader, sampled_data, count):
    assert len(sampled_data) == count
    assert sampled_data.columns.tolist() == dummy_single_table_data_loader.columns()


def test_ctgan(
    ctgan: CTGANSynthesizerModel,
    dummy_single_table_metadata,
    dummy_single_table_data_loader,
    save_model_dir,
):
    ctgan.fit(dummy_single_table_metadata, dummy_single_table_data_loader)
    sampled_data = ctgan.sample(10)
    assert_sampled_data(dummy_single_table_data_loader, sampled_data, 10)

    ctgan.save(save_model_dir)
    assert save_model_dir.exists()

    model = CTGANSynthesizerModel.load(save_model_dir)
    sampled_data = model.sample(10)
    assert_sampled_data(dummy_single_table_data_loader, sampled_data, 10)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
