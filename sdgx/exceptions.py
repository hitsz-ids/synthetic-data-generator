class SdgxError(Exception):
    """
    Base class for exceptions in this module.
    """

    EXIT_CODE = 100
    ERROR_CODE = 1001


class NonParametricError(Exception):
    """
    Exception to indicate that a model is not parametric.
    """

    EXIT_CODE = 101
    ERROR_CODE = 2001


class ManagerError(SdgxError):
    """
    Exception to indicate that exception when using manager.
    """

    EXIT_CODE = 102
    ERROR_CODE = 3000


class NotFoundError(ManagerError):
    """
    Exception to indicate that a model is not found.
    """

    ERROR_CODE = 3001


class RegisterError(ManagerError):
    """
    Exception to indicate that exception when registering.
    """

    ERROR_CODE = 3002


class InitializationError(ManagerError):
    """
    Exception to indicate that exception when initializing model.
    """

    ERROR_CODE = 3003


class ManagerLoadModelError(ManagerError):
    """
    Exception to indicate that exception when loading model for :ref:`ModelManager`.
    """

    ERROR_CODE = 3004


class SynthesizerError(SdgxError):
    """
    Exception to indicate that exception when synthesizing model.
    """

    EXIT_CODE = 103
    ERROR_CODE = 4000


class SynthesizerInitError(SynthesizerError):
    ERROR_CODE = 4001


class SynthesizerSampleError(SynthesizerError):
    ERROR_CODE = 4002


class SynthesizerProcessorError(SynthesizerError):
    ERROR_CODE = 4003


class CacheError(SdgxError):
    """
    Exception to indicate that exception when using cache.
    """

    EXIT_CODE = 104
    ERROR_CODE = 5001


class MetadataInitError(SdgxError):
    """
    Exception to indicate that exception when initializing metadata.
    """

    EXIT_CODE = 105
    ERROR_CODE = 6001


class DataLoaderInitError(SdgxError):
    """
    Exception to indicate that exception when initializing dataloader.
    """

    EXIT_CODE = 106
    ERROR_CODE = 7001


class CannotExportError(SdgxError):
    """
    Exception to indicate that exception when exporting data.
    """

    EXIT_CODE = 107
    ERROR_CODE = 8001


class DataModelError(SdgxError):
    """
    Exception to indicate that exception in all data models.
    """

    EXIT_CODE = 108
    ERROR_CODE = 9001


class MetadataInvalidError(DataModelError):
    ERROR_CODE = 9002


class RelationshipInitError(DataModelError):
    ERROR_CODE = 9003


class MetadataCombinerError(DataModelError):
    ERROR_CODE = 9004


class MetadataCombinerInvalidError(MetadataCombinerError):
    ERROR_CODE = 9005


class MetadataCombinerInitError(MetadataCombinerError):
    ERROR_CODE = 9006


class InspectorInitError(DataModelError):
    ERROR_CODE = 9007
