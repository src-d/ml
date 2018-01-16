import operator

from pyspark import Row

from sourced.ml.transformers import Transformer


class Uast2DocFreq(Transformer):

    def __init__(self, extractors, document_column, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.document_column = document_column
        self.ndocs = None

    def __call__(self, rows):
        processed = rows.flatMap(self._process_row)
        if self.explained:
            self._log.info("toDebugString():\n%s", processed.toDebugString().decode())
        self.ndocs = rows.map(lambda x: getattr(x, self.document_column)).distinct().count()
        return processed \
            .distinct() \
            .map(lambda x: (x[0], 1)) \
            .reduceByKey(operator.add) \
            .map(lambda x: Row(token=x[0], value=x[1]))

    def _process_row(self, row):
        document = getattr(row, self.document_column)
        for extractor in self.extractors:
            for key, val in extractor.extract(row.uast):
                yield key, document


class Uast2Quant(Transformer):
    def __init__(self, extractors, nb_partitions, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.nb_partitions = nb_partitions

    def __call__(self, rows):
        for i, extractor in enumerate(self.extractors):
            try:
                quantize = extractor.quantize
            except AttributeError:
                self._log.debug("%s: no quantization performed", extractor.__class__.__name__)
                continue
            self._log.info("%s: performing quantization with %d partitions",
                           extractor.__class__.__name__, self.nb_partitions)
            all_children = rows.flatMap(lambda j: self.process_row(j, extractor))
            all_children_reduced = all_children.countByKey()
            children_freq = extractor._get_children_freqs(all_children_reduced)
            quantize(children_freq, self.nb_partitions)

    def process_row(self, row, extractor):
        for k in extractor.inspect_quant(row.uast):
            yield k, 1
