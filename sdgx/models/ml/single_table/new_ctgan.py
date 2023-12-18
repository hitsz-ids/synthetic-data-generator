from sdgx.models.extension import hookimpl
from sdgx.models.ml.single_table.base import MLSynthesizerModel


class CTGANSynthesizerModel(MLSynthesizerModel):
    pass


# @hookimpl
# def register(manager):
#     manager.register("CTGAN", CTGANSynthesizerModel)
