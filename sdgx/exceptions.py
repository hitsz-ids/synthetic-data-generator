class SdgxError(Exception):
    """
    Base class for exceptions in this module.
    """


class NonParametricError(Exception):
    """
    Exception to indicate that a model is not parametric.
    """


class ModelNotFoundError(SdgxError):
    """
    Exception to indicate that a model is not found.
    """


class ModelRegisterError(SdgxError):
    """
    Exception to indicate that exception when registering model.
    """


class ModelInitializationError(SdgxError):
    """
    Exception to indicate that exception when initializing model.
    """
