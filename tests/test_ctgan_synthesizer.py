import pytest

from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer


@pytest.fixture
def ctgan_synthesizer(demo_single_table_data_pos_neg_connector):
    yield Synthesizer(
        model=CTGANSynthesizerModel(epochs=1),
        data_connector=demo_single_table_data_pos_neg_connector,
    )


def test_ctgan_synthesizer_with_pos_neg(
    ctgan_synthesizer: Synthesizer,
    demo_single_table_data_pos_neg_metadata,
    demo_single_table_data_pos_neg_loader,
    demo_single_table_data_pos_neg,
):
    original_data = demo_single_table_data_pos_neg
    metadata = demo_single_table_data_pos_neg_metadata

    # Train the CTGAN model
    ctgan_synthesizer.fit(
        demo_single_table_data_pos_neg_metadata, demo_single_table_data_pos_neg_loader
    )
    sampled_data = ctgan_synthesizer.sample(1000)

    # Check each column
    for column in original_data.columns:
        # Skip columns that are identifiers or not intended for positivity checks
        if column == "int_id":
            continue

        is_all_positive = (original_data[column] >= 0).all()
        is_all_negative = (original_data[column] <= 0).all()

        if is_all_positive:
            # Assert that the sampled_data column is also all positive
            assert (
                sampled_data[column] >= 0
            ).all(), f"Column '{column}' in sampled data should be all positive."
        elif is_all_negative:
            # Assert that the sampled_data column is also all negative
            assert (
                sampled_data[column] <= 0
            ).all(), f"Column '{column}' in sampled data should be all negative."


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
