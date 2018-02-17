import operator
from typing import Iterable

from sourced.ml.extractors import BagsExtractor
from sourced.ml.transformers import Transformer


class Uast2Quant(Transformer):
    def __init__(self, extractors: Iterable[BagsExtractor], **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors

    def __call__(self, rows):
        for i, extractor in enumerate(self.extractors):
            try:
                quantize = extractor.quantize
            except AttributeError:
                self._log.debug("%s: no quantization performed", extractor.__class__.__name__)
                continue
            self._log.info("%s: performing quantization with %d partitions",
                           extractor.__class__.__name__, extractor.npartitions)
            items = rows \
                .flatMap(lambda row: extractor.extract(row.uast)) \
                .reduceByKey(operator.add) \
                .map(lambda x: (x[0][0], (x[0][1], x[1]))) \
                .groupByKey().mapValues(list).collect()
            # (x[0][0], (x[0][1], x[1]))) <=> (class, (instance, frequency))
            quantize(items)
            self._log.info("Done")
