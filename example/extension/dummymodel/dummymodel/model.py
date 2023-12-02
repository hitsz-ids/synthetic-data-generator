from __future__ import annotations

from sdgx.models.base import BaseSynthesizerModel


class MyOwnModel(BaseSynthesizerModel):
    ...


from sdgx.models.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("DummyModel", MyOwnModel)
