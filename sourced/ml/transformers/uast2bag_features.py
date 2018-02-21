from typing import Iterable, Union

from pyspark import RDD, Row

from sourced.ml.extractors import BagsExtractor
from sourced.ml.transformers import Transformer


class Uast2BagFeatures(Transformer):
    def __init__(self, extractors: Iterable[BagsExtractor],
                 document_column: Union[str, int], **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.document_column = document_column

    def __call__(self, rows: RDD):
        return rows.flatMap(self.process_row)

    def process_row(self, row: Row):
        doc = getattr(row, self.document_column)
        for extractor in self.extractors:
            for key, val in extractor.extract(row.uast):
                yield (key, doc), val
