from sourced.ml.transformers import Transformer


class Repo2DocFreq(Transformer):
    NDOCS_KEY = -1, 0

    def __init__(self, extractors, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors

    def __call__(self, rows):
        processed = rows.flatMap(self.process_row)
        if self.explained:
            self._log.info("toDebugString():\n%s", processed.toDebugString().decode())
        reduced = processed.countByKey()
        ndocs = None
        for (i, key), value in reduced.items():
            if (i, key) == self.NDOCS_KEY:
                ndocs = value
                continue
            self.extractors[i].apply_docfreq(key, value)

        for extractor in self.extractors:
            extractor.ndocs = ndocs

    def process_row(self, row):
        yield self.NDOCS_KEY, 1
        for i, extractor in enumerate(self.extractors):
            for k in extractor.inspect(row.uast):
                yield (i, k), 1


class Repo2Quant(Transformer):

    def __init__(self, extractors, nb_partitions, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.nb_partitions = nb_partitions

    def __call__(self, rows):
        for i, extractor in enumerate(self.extractors):
            if "build_quantization" in dir(extractor):
                self._log.info("Perform quantization with %d partitions", self.nb_partitions)
                all_children = rows.flatMap(lambda j: self.process_row(j, extractor))
                all_children_reduced = all_children.countByKey()
                children_freq = extractor.get_children_freq(all_children_reduced)
                extractor.build_quantization(children_freq, self.nb_partitions)

    def process_row(self, row, extractor):
        for k in extractor.inspect_quant(row.uast):
            yield k, 1
