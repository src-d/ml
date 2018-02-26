from typing import Iterable, Union

from pyspark import RDD, Row
from pyspark.sql import DataFrame

from sourced.ml.extractors import BagsExtractor
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


class Uast2BagFeatures(Transformer):
    class Columns:
        """
        Standard column names for interop.
        """
        token = "token"
        document = "document"
        value = "value"

    def __init__(self, extractors: Iterable[BagsExtractor], **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors

    def __call__(self, rows: RDD):
        return rows.flatMap(self.process_row)

    def process_row(self, row: Row):
        uast_column = EngineConstants.Columns.Uast
        doc = row[self.Columns.document]
        for uast in row[uast_column]:
            for extractor in self.extractors:
                for key, val in extractor.extract(uast):
                    yield (key, doc), val
