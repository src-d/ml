from itertools import chain

from pyspark import Row, RDD
from pyspark.sql import DataFrame

from sourced.ml.transformers import Transformer
from sourced.ml.utils import EngineConstants


class Moder(Transformer):
    """
    Select the items to extract from UASTs.
    """
    class Options:
        repo = "repo"
        file = "file"
        function = "func"

        __all__ = (file, function, repo)

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
            data[EngineConstants.Columns.Uast] = [self.serialize_uast(func)]
            data[EngineConstants.Columns.BlobId] += "_%s:%d" % (name, func.start_position.line)
            yield Row(**data)

    def extract_functions_from_uast(self, uast: "bblfsh.Node"):
        allfuncs = list(self.filter_uast(uast, self.FUNC_XPATH))
        internal = set()
        for func in allfuncs:
            if id(func) in internal:
                continue
            for sub in self.filter_uast(func, self.FUNC_XPATH):
                if sub != func:
                    internal.add(id(sub))
        for f in allfuncs:
            if id(f) not in internal:
                name = "+".join(n.token for n in self.filter_uast(f, self.FUNC_NAME_XPATH))
                if name:
                    yield f, name
