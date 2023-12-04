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

    We provided an example package for you in {project_root}/example/extension/dummymodel.

    Example:
    .. code-block:: python

        class MyOwnModel(BaseSynthesizerModel):
            ...

        from sdgx.models.extension import hookimpl

        @hookimpl
        def register(manager):
            manager.register("DummyModel", MyOwnModel)


    Config ``project.entry-points`` so that we can find it

    .. code-block:: toml
    [project.entry-points."sdgx.model"]
    < whatever-name > = "<package>.<path>.<to>.<model-file>"


    You can verify it by `sdgx list-models`.
    """
