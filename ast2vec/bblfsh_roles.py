from bblfsh.github.com.bblfsh.sdk.uast.generated_pb2 import DESCRIPTOR


def _get_role_id(role_name):
    return DESCRIPTOR.enum_types_by_name["Role"].values_by_name[role_name].number


SIMPLE_IDENTIFIER = _get_role_id("SIMPLE_IDENTIFIER")
FUNCTION_DECLARATION = _get_role_id("FUNCTION_DECLARATION")
FUNCTION_DECLARATION_NAME = _get_role_id("FUNCTION_DECLARATION_NAME")
IMPORT_ALIAS = _get_role_id("IMPORT_ALIAS")
IMPORT_PATH = _get_role_id("IMPORT_PATH")
CALL_CALLEE = _get_role_id("CALL_CALLEE")
CALL = _get_role_id("CALL")
