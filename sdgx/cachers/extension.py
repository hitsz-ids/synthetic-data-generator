from __future__ import annotations

import pluggy

project_name = "sdgx.cacher"
hookimpl = pluggy.HookimplMarker(project_name)
hookspec = pluggy.HookspecMarker(project_name)


@hookspec
def register(manager):
    """
    For more information about this function, please check the :ref:`Manager`

    We provided an example package for you in {project_root}/example/extension/dummycacher.

    Example:
    .. code-block:: python

        class MyOwnCache(Cacher):
            ...

        from sdgx.cachers.extension import hookimpl

        @hookimpl
        def register(manager):
            manager.register("DummyDataCacher", MyOwnCache)


    Config ``project.entry-points`` so that we can find it

    .. code-block:: toml
    [project.entry-points."sdgx.cacher"]
    < whatever-name > = "<package>.<path>.<to>.<class-file>"


    You can verify it by `sdgx list-cacher`.
    """
