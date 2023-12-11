from sdgx.data_processors.base import DataProcessor


class Sampler(DataProcessor):
    """
    Base class for samplers.

    Sampler is used to reduce the size of data against the large scale dataset.

    Some models may embed sampler in the model.
    """
