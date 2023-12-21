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


def test_ctgan(
    ctgan: CTGANSynthesizerModel,
    demo_single_table_metadata,
    demo_single_table_data_loader,
    save_model_dir,
):
    ctgan.fit(demo_single_table_metadata, demo_single_table_data_loader)
    ctgan.sample(10)
    ctgan.save(save_model_dir)
    model = CTGANSynthesizerModel.load(save_model_dir)
    model.sample(10)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
