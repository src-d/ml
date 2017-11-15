import itertools

import numpy
from pyspark.sql.types import Row

from sourced.ml.repo2.base import Transformer
from sourced.ml.uast_ids_to_bag import UastIds2Bag


__extractors__ = {}


def register_extractor(cls):
    if not issubclass(cls, BagsExtractor):
        raise TypeError("%s is not an instance of %s" % (cls.__name__, BagsExtractor.__name__))
    __extractors__[cls.NAME] = cls
    return cls


class BagsExtractor:
    DEFAULT_DOCFREQ_THRESHOLD = 5

    def __init__(self, docfreq_threshold=None):
        if docfreq_threshold is None:
            docfreq_threshold = self.DEFAULT_DOCFREQ_THRESHOLD
        self.docfreq_threshold = docfreq_threshold
        self.docfreq = {}
        self._ndocs = 0

    @property
    def docfreq_threhold(self):
        return self._docfreq_threshold

    @docfreq_threhold.setter
    def docfreq_threshold(self, value):
        if not isinstance(value, int):
            raise TypeError("docfreq_threshold must be an integer, got %s" % type(value))
        if value < 1:
            raise ValueError("docfreq_threshold must be >= 1, got %d" % value)
        self._docfreq_threshold = value

    @property
    def ndocs(self):
        return self._ndocs

    @ndocs.setter
    def ndocs(self, value):
        if not isinstance(value, int):
            raise TypeError("ndocs must be an integer, got %s" % type(value))
        if value < 1:
            raise ValueError("ndocs must be >= 1, got %d" % value)
        self._ndocs = value

    def extract(self, uast):
        raise NotImplementedError()

    def inspect(self, uast):
        raise NotImplementedError()

    def apply_docfreq(self, key, value):
        if value >= self.docfreq_threshold:
            if not isinstance(key, str):
                raise TypeError("key is %s" % type(key))
            self.docfreq[str(key)] = value


class Repo2WeightedSet(Transformer):
    def __init__(self, extractors, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors

    def __call__(self, rows):
        return rows.flatMap(self.process_row)

    def process_row(self, row):
        bag = {}
        for extractor in self.extractors:
            bag.update(extractor.extract(row.uast))
        yield row.file_hash, bag


class Repo2DocFreq(Transformer):
    NDOCS_KEY = -1, 0

    def __init__(self, extractors, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors

    def __call__(self, rows):
        processed = rows.flatMap(self.process_row)
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


@register_extractor
class IdentifiersBagExtractor(BagsExtractor):
    NAME = "id"

    class NoopTokenParser:
        def process_token(self, token):
            yield token

    def __init__(self, docfreq_threshold=None, split_stem=False):
        super().__init__(docfreq_threshold)
        self.id2bag = UastIds2Bag(
            None, self.NoopTokenParser() if not split_stem else None)

    def extract(self, uast):
        ndocs = self.ndocs
        docfreq = self.docfreq
        log = numpy.log
        for key, val in self.id2bag.uast_to_bag(uast).items():
            try:
                yield key, log(1 + val) * log(ndocs / docfreq[key])
            except KeyError:
                # docfreq_threshold
                continue

    def inspect(self, uast):
        try:
            bag = self.id2bag.uast_to_bag(uast)
        except RuntimeError as e:
            raise ValueError(str(uast)) from e
        for key in bag:
            yield key


class BagsBatcher(Transformer):
    DEFAULT_CHUNK_SIZE = 1.5 * 1000 * 1000 * 1000

    def __init__(self, extractors, chunk_size=DEFAULT_CHUNK_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.keys = {}
        self.chunk_size = chunk_size

    def __call__(self, processed):
        avglen = processed.values().map(len).mean()
        ndocs = self.extractors[0].ndocs
        self._log.info("Average bag length: %.1f", avglen)
        self._log.info("Number of documents: %d", ndocs)
        keys = list(itertools.chain.from_iterable(e.docfreq for e in self.extractors))
        keys.sort()
        self.keys = {k: i for i, k in enumerate(keys)}
        del keys
        chunklen = int(self.chunk_size / (2 * 4 * avglen))
        nparts = ndocs // chunklen + 1
        chunklen = int(ndocs / nparts * (2 * 4 * avglen))
        self._log.info("chunk %d\tparts %d", chunklen, nparts)
        return processed.mapValues(self.bag2row).repartition(nparts)

    def bag2row(self, bag):
        data = numpy.zeros(len(bag), dtype=numpy.float32)
        indices = numpy.zeros(len(bag), dtype=numpy.int32)
        for i, (k, v) in enumerate(sorted((self.keys[k], v) for k, v in bag.items())):
            data[i] = v
            indices[i] = k
        return data, indices


class BagsBatchSaver(Transformer):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def __call__(self, head):
        self._log.info("Writing to %s", self.path)
        head \
            .mapPartitions(self.concatenate) \
            .map(lambda x: Row(**x)) \
            .toDF() \
            .write.parquet(self.path)

    def concatenate(self, part):
        data = []
        indices = []
        indptr = [0]
        keys = []
        for k, (d, i) in part:
            keys.append(k)
            data.append(d)
            indices.append(i)
            indptr.append(indptr[-1] + d.shape[0])
        data = numpy.concatenate(data).astype(dtype=numpy.float32)
        indices = numpy.concatenate(indices).astype(dtype=numpy.int32)
        indptr = numpy.array(indptr, dtype=numpy.int64)
        return [{"keys": bytearray("\0".join(keys).encode()),
                 "data": bytearray(data.data),
                 "indices": bytearray(indices.data),
                 "indptr": bytearray(indptr.data),
                 "rows": indptr.shape[0]}]
