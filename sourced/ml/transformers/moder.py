from itertools import chain
import os

from pyspark import Row, RDD
from pyspark.sql import DataFrame

from sourced.ml.algorithms.uast_ids_to_bag import uast2sequence
from sourced.ml.transformers import Transformer
from sourced.ml.utils import EngineConstants, FUNCTION, DECLARATION, NAME, IDENTIFIER


class Moder(Transformer):
    """
    Select the items to extract from UASTs.
    """
    class Options:
        repo = "repo"
        file = "file"
        function = "func"

        __all__ = (file, function, repo)

    USE_XPATH = os.getenv("USE_XPATH", False) in ("1", "true", "yes")

    # Copied from https://github.com/src-d/hercules/blob/master/shotness.go#L40
    # If you change here, please PR it to Hercules as well
    FUNC_XPATH = "//*[@roleFunction and @roleDeclaration]"
    FUNC_NAME_XPATH = "/*[@roleFunction and @roleIdentifier and @roleName] " \
                      "| /*/*[@roleFunction and @roleIdentifier and @roleName]"

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def __setstate__(self, state):
        super().__setstate__(state)
        from bblfsh import Node, filter as filter_uast
        self.parse_uast = Node.FromString
        self.serialize_uast = Node.SerializeToString
        self.filter_uast = filter_uast

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if not isinstance(value, str):
            raise TypeError("mode must be a string")
        if value not in self.Options.__all__:
            raise ValueError("Unsupported mode: " + value)
        self._mode = value

    def call_repo(self, rows: RDD):
        ridcol = EngineConstants.Columns.RepositoryId
        uastcol = EngineConstants.Columns.Uast
        return rows \
            .groupBy(lambda r: r[ridcol]) \
            .map(lambda x: Row(**{ridcol: x[0], EngineConstants.Columns.Path: "",
                                  EngineConstants.Columns.BlobId: "",
                                  uastcol: list(chain.from_iterable(i[uastcol] for i in x[1]))}))

    def call_file(self, rows: RDD):
        return rows

    def call_func(self, rows: RDD):
        return rows.flatMap(self.extract_functions_from_row)

    def __call__(self, rows: DataFrame) -> RDD:
        return getattr(self, "call_" + self.mode)(rows.rdd)

    def extract_functions_from_row(self, row: Row):
        uastbytes = row[EngineConstants.Columns.Uast]
        if not uastbytes:
            return
        uast = self.parse_uast(uastbytes[0])
        template = row.asDict()
        for func, name in self.extract_functions_from_uast(uast):
            data = template.copy()
            data[EngineConstants.Columns.Uast] = [bytearray(self.serialize_uast(func))]
            data[EngineConstants.Columns.BlobId] += "_%s:%d" % (name, func.start_position.line)
            yield Row(**data)

    def extract_functions_from_uast(self, uast: "bblfsh.Node"):
        if self.USE_XPATH:
            allfuncs = list(self.filter_uast(uast, self.FUNC_XPATH))
        else:
            node_seq = uast2sequence(uast)
            allfuncs = [node for node in node_seq if FUNCTION in node.roles and
                        DECLARATION in node.roles]
        internal = set()
        for func in allfuncs:
            if id(func) in internal:
                continue

            if self.USE_XPATH:
                sub_seq = self.filter_uast(func, self.FUNC_XPATH)
            else:
                sub_seq = [node for node in uast2sequence(func) if FUNCTION in node.roles and
                           DECLARATION in node.roles]

            for sub in sub_seq:
                if sub != func:
                    internal.add(id(sub))
        for f in allfuncs:
            if id(f) not in internal:
                if self.USE_XPATH:
                    f_seq = self.filter_uast(f, self.FUNC_NAME_XPATH)
                else:
                    f_seq = [node for node in uast2sequence(f) if FUNCTION in node.roles and
                             IDENTIFIER in node.roles and NAME in node.roles]
                name = "+".join(n.token for n in f_seq)
                if name:
                    yield f, name
