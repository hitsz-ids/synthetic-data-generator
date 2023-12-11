class Transformer:
    """
    Base class for transformers.

    Transformer is used to transform table data from one format to another.
    For example, encode discrete column into one hot encoding.

    To achieve that, Transformer can use :ref:`Formatter` and :ref:`Inspector` to help.
    """
