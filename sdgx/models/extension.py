from __future__ import annotations

from typing import Any

import pluggy

project_name = "sdgx.model"
hookimpl = pluggy.HookimplMarker(project_name)
hookspec = pluggy.HookspecMarker(project_name)


@hookspec
def register(manager):
    """
    For more information about this function, please check the ``ModelManager``

    Example:
    .. code-block:: python

        class MyOwnModel(BaseSynthesizerModel):
            ...

        from sdgx.models.extension import hookimpl

        @hookimpl
        def register(manager):
            manager.register("DummyModel", MyOwnModel)
    """
