from __future__ import annotations

from sdgx.cachers.base import Cacher


class MyOwnCache(Cacher): ...


from sdgx.cachers.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("DummyCache", MyOwnCache)
