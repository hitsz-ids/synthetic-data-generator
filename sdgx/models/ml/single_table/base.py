from sdgx.models.base import SynthesizerModel


class MLSynthesizerModel(SynthesizerModel):
    """
    Base class for ML models
    """

    fit_data_empty: bool = False
