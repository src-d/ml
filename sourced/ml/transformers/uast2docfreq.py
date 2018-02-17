import operator
from typing import Iterable, Union

from pyspark import Row

from sourced.ml.extractors import BagsExtractor
from sourced.ml.transformers import Transformer


class Uast2DocFreq(Transformer):
    def __init__(self, extractors: Iterable[BagsExtractor],
                 document_column: Union[str, int], **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.document_column = document_column
        self.ndocs = None

    def __call__(self, rows):
        processed = rows.flatMap(self.process_row)
        self._log.info("Calculating the number of distinct documents...")
        if self.explained:
            self._log.info("toDebugString():\n%s", processed.toDebugString().decode())
        self.ndocs = rows.map(lambda x: getattr(x, self.document_column)).distinct().count()
        self._log.info("Done")
        return processed \
            .distinct() \
            .map(lambda x: (x[0], 1)) \
            .reduceByKey(operator.add) \
            .map(lambda x: Row(token=x[0], value=x[1]))

    def process_row(self, row):
        document = getattr(row, self.document_column)
        for extractor in self.extractors:
            for key, val in extractor.extract(row.uast):
                yield key, document
