from typing import Iterable, Union

from pyspark import RDD, Row
from pyspark.sql import DataFrame

from sourced.ml.extractors import BagsExtractor
from sourced.ml.transformers.transformer import Transformer
from sourced.ml.utils import EngineConstants


class UastRow2Document(Transformer):
    def __call__(self, rows: Union[RDD, DataFrame]):
        if isinstance(rows, DataFrame):
            rows = rows.rdd
        return rows.map(self.documentize)

    def documentize(self, r: Row) -> Row:
        ec = EngineConstants.Columns
        doc = "%s/%s@%s" % (r[ec.RepositoryId], r[ec.Path], r[ec.BlobId])
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
        for extractor in self.extractors:
            for key, val in extractor.extract(row[uast_column]):
                yield (key, doc), val
