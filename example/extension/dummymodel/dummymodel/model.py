from __future__ import annotations

from sdgx.models.base import SynthesizerModel


class MyOwnModel(SynthesizerModel): ...


from sdgx.models.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("DummyModel", MyOwnModel)
