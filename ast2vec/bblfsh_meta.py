import importlib

import ast2vec.lazy_grpc as lazy_grpc

with lazy_grpc.masquerade():
    # All the stuff below does not really need grpc so we arrange the delayed import.

    from bblfsh.sdkversion import VERSION

    Node = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION).Node

    _ROLE = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION)._ROLE
    for desc in _ROLE.values:
        globals()[desc.name] = desc.index
    del _ROLE
