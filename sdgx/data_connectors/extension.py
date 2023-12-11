from __future__ import annotations

import pluggy

project_name = "sdgx.data_connector"
hookimpl = pluggy.HookimplMarker(project_name)
hookspec = pluggy.HookspecMarker(project_name)


@hookspec
def register(manager):
    """
    For more information about this function, please check the :ref:`Manager`

    We provided an example package for you in {project_root}/example/extension/dummydataconnector.

    Example:
    .. code-block:: python

        class MyOwnDataConnector(DataConnector):
            ...

        from sdgx.data_connectors.extension import hookimpl

        @hookimpl
        def register(manager):
            manager.register("DummyDataConnector", MyOwnDataConnector)


    Config ``project.entry-points`` so that we can find it

    .. code-block:: toml
    [project.entry-points."sdgx.data_connector"]
    < whatever-name > = "<package>.<path>.<to>.<class-file>"


    You can verify it by `sdgx list-data-connectors`.
    """
