import importlib

import ast2vec.lazy_grpc as lazy_grpc

with lazy_grpc.masquerade():
    # All the stuff below does not really need grpc so we arrange the delayed import.

    from bblfsh.sdkversion import VERSION

    DESCRIPTOR = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION).DESCRIPTOR
    Node = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION).Node

    def _get_role_id(role_name):
        return DESCRIPTOR.enum_types_by_name["Role"].values_by_name[role_name].number

    SIMPLE_IDENTIFIER = _get_role_id("SIMPLE_IDENTIFIER")
    FUNCTION_DECLARATION = _get_role_id("FUNCTION_DECLARATION")
    FUNCTION_DECLARATION_NAME = _get_role_id("FUNCTION_DECLARATION_NAME")
    FUNCTION_DECLARATION_BODY = _get_role_id("FUNCTION_DECLARATION_BODY")
    IMPORT_ALIAS = _get_role_id("IMPORT_ALIAS")
    IMPORT_PATH = _get_role_id("IMPORT_PATH")
    CALL_CALLEE = _get_role_id("CALL_CALLEE")
    CALL = _get_role_id("CALL")
    TYPE_DECLARATION = _get_role_id("TYPE_DECLARATION")
    ASSIGNMENT_VARIABLE = _get_role_id("ASSIGNMENT_VARIABLE")
    QUALIFIED_IDENTIFIER = _get_role_id("QUALIFIED_IDENTIFIER")
