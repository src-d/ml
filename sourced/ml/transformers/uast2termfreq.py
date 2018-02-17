import operator
from typing import Iterable, Union

from pyspark import Row

from sourced.ml.extractors import BagsExtractor
from sourced.ml.transformers import Transformer


class Uast2TermFreq(Transformer):
    def __init__(self, extractors: Iterable[BagsExtractor],
                 document_column: Union[str, int], **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.document_column = document_column

    def __call__(self, rows):
        return rows \
            .flatMap(self.process_row) \
            .reduceByKey(operator.add) \
            .map(lambda x: Row(document=x[0][0], token=x[0][1], value=x[1]))

    def process_row(self, row):
        doc_id = getattr(row, self.document_column)
        for extractor in self.extractors:
            for key, val in extractor.extract(row.uast):
                yield (doc_id, key), val
