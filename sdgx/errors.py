# TBD
# 主要用于存放 sdg 中特有的的报错信息


class SdgxError(Exception):
    """Base class for exceptions in this module."""

    pass


class NonParametricError(Exception):
    """Exception to indicate that a model is not parametric."""


class ModelNotFoundError(SdgxError):
    pass


class ModelRegisterError(SdgxError):
    pass


class ModelInitializationError(SdgxError):
    pass
