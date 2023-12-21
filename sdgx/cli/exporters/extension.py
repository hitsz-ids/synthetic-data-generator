from __future__ import annotations

import pluggy

project_name = "sdgx.cli.exporter"
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

    We provided an example package for you in ``{project_root}/example/extension/dummyexporter``.

    Example:

    .. code-block:: python

        class MyOwnExporter(Exporter):
            ...

        from sdgx.cli.exporters.extension import hookimpl

        @hookimpl
        def register(manager):
            manager.register("MyOwnExporter", MyOwnExporter)


    Config ``project.entry-points`` so that we can find it

    .. code-block:: toml

        [project.entry-points."sdgx.cli.exporter"]
        {whatever-name} = "{package}.{path}.{to}.{file-with-hookimpl-function}"


    You can verify it by `sdgx list-data-connectors`.
    """
