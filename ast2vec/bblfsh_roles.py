
from bblfsh.github.com.bblfsh.sdk.uast.generated_pb2 import DESCRIPTOR


SIMPLE_IDENTIFIER = DESCRIPTOR.enum_types_by_name["Role"] \
    .values_by_name["SIMPLE_IDENTIFIER"].number
