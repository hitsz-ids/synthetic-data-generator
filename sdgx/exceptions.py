class SdgxError(Exception):
    """
    Base class for exceptions in this module.
    """


class NonParametricError(Exception):
    """
    Exception to indicate that a model is not parametric.
    """


class ManagerError(SdgxError):
    """
    Exception to indicate that exception when using manager.
    """


class NotFoundError(ManagerError):
    """
    Exception to indicate that a model is not found.
    """


class RegisterError(ManagerError):
    """
    Exception to indicate that exception when registering.
    """


class InitializationError(ManagerError):
    """
    Exception to indicate that exception when initializing model.
    """


class ManagerLoadModelError(ManagerError):
    """
    Exception to indicate that exception when loading model for :ref:`ModelManager`.
    """


class SynthesizerInitError(ManagerError):
    """
    Exception to indicate that exception when synthesizing model.
    """


class CacheError(SdgxError):
    """
    Exception to indicate that exception when using cache.
    """
