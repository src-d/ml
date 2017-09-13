from contextlib import contextmanager
import sys


class LazyGrpc:
    """
    This class serves two functions.
    First, it has masquerade() which is the context manager which
    imitates the real grpc module and cleans up at exit.
    Second, it delegates all the unresolved attributes to the
    real grpc module.
    Thus this class is useful for importing modules which depend
    on grpc without actually importing grpc. As soon as there is
    a real need in grpc's internals, the delayed import happens.
    """
    def __init__(self):
        self.__dict__ = sys.modules[__name__].__dict__
        self.__grpc = None

    @contextmanager
    def masquerade(self):
        assert "grpc" not in sys.modules
        sys.modules["grpc"] = sys.modules[__name__]
        yield None
        del sys.modules["grpc"]

    def __getattr__(self, item):
        if self.__grpc is None:
            import grpc
            self.__grpc = grpc
        return getattr(self.__grpc, item)


sys.modules[__name__] = LazyGrpc()
