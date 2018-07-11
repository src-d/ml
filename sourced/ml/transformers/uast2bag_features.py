from typing import Iterable, Union

from pyspark import RDD, Row
from pyspark.sql import DataFrame

from sourced.ml.extractors import Extractor
from sourced.ml.transformers.transformer import Transformer
from sourced.ml.utils import EngineConstants


class UastRow2Document(Transformer):
    REPO_PATH_SEP = "//"
    PATH_BLOB_SEP = "@"

    def __call__(self, rows: Union[RDD, DataFrame]):
        if isinstance(rows, DataFrame):
            rows = rows.rdd
        return rows.map(self.documentize)

    def documentize(self, r: Row) -> Row:
        ec = EngineConstants.Columns
        doc = r[ec.RepositoryId]
        if r[ec.Path]:
            doc += self.REPO_PATH_SEP + r[ec.Path]
        if r[ec.BlobId]:
            doc += self.PATH_BLOB_SEP + r[ec.BlobId]
        bfc = Uast2BagFeatures.Columns
        return Row(**{bfc.document: doc, ec.Uast: r[ec.Uast]})


class Uast2Features(Transformer):
    def __init__(self, extractors: Union[Extractor, Iterable[Extractor]], **kwargs):
        super().__init__(**kwargs)
        self.extractors = [extractors] if isinstance(extractors, Extractor) else extractors

    def __call__(self, rows: RDD):
        return rows.flatMap(self.process_row)

    def _to_result(self, row: Row, name, feature):
        new = row.asDict()
        new[name] = feature
        return new

    def process_row(self, row: Row):
        for uast in row[EngineConstants.Columns.Uast]:
            for extractor in self.extractors:
                for feature in extractor.extract(uast):
                    yield self._to_result(row, extractor.NAME, feature)


class Uast2BagFeatures(Uast2Features):
    class Columns:
        """
        Standard column names for interop.
        """
        token = "token"
        document = "document"
        value = "value"

    def _to_result(self, row: Row, name, feature):
        return (feature[0], row[self.Columns.document]), feature[1]
