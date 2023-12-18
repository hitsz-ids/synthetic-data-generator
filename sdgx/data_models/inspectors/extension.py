from __future__ import annotations

from typing import Any

import pluggy

project_name = "sdgx.metadata.inspector"
"""
The entry-point name of this extension.

Should be used in ``pyproject.toml`` as ``[project.entry-points."{project_name}"]``
"""
hookimpl = pluggy.HookimplMarker(project_name)
"""
Hookimpl marker for this extension, extension module should use this marker

Example:

    .. code-block:: python

        @hookimpl
        def register(manager):
            ...
"""

hookspec = pluggy.HookspecMarker(project_name)


@hookspec
def register(manager):
    """
    For more information about this function, please check the :ref:`manager`

    We provided an example package for you in ``{project_root}/example/extension/dummymetadatainspector``.

    Example:

    .. code-block:: python

        class MyOwnInspector(Inspector):
            ...

        from sdgx.data_models.inspectors.extension import hookimpl

        @hookimpl
        def register(manager):
            manager.register("DummyInspector", MyOwnInspector)


    Config ``project.entry-points`` so that we can find it

    .. code-block:: toml

        [project.entry-points."sdgx.metadata.inspector"]
        {whatever-name} = "{package}.{path}.{to}.{file-with-hookimpl-function}"
    """
