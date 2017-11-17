from collections import namedtuple
import itertools
from glob import glob
import os
from typing import Iterable, Dict

from modelforge import merge_strings, register_model
import numpy
import parquet
from pyspark.sql.types import Row
from scipy.sparse import csr_matrix

from sourced.ml.df import DocumentFrequencies
from sourced.ml.pickleable_logger import PickleableLogger
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


@register_model
class OrderedDocumentFrequencies(DocumentFrequencies):
    """
    Compatible with the original DocumentFrequencies.
    """
    NAME = "ordered_docfreq"

    def construct(self, docs: int, dicts: Iterable[Dict[str, int]]):
        df = {}
        for d in dicts:
            df.update(d)
        super().construct(docs, df)
        self._log.info("Ordering the keys...")
        keys = list(self._df)
        keys.sort()
        self._order = {k: i for i, k in enumerate(keys)}
        return self

    @property
    def order(self):
        return self._order

    def _load_tree(self, tree):
        tokens = None
        original_construct = self.construct
        super_construct = super().construct

        def hacked_construct(docs, tokfreq, **kwargs):
            super_construct(docs=docs, tokfreq=tokfreq)
            nonlocal tokens
            tokens = kwargs["tokens"]

        self.construct = hacked_construct
        try:
            super()._load_tree(tree)
        finally:
            self.construct = original_construct
        self._log.info("Mapping the keys order...")
        self._order = {k: i for i, k in enumerate(tokens)}

    def _generate_tree(self):
        tokens = [None] * len(self)
        freqs = numpy.zeros(len(self), dtype=numpy.float32)
        for k, i in self._order.items():
            tokens[i] = k
            freqs[i] = self._df[k]
        return {"docs": self.docs, "tokens": merge_strings(tokens), "freqs": freqs}


class BagsBatcher(Transformer):
    DEFAULT_CHUNK_SIZE = 1.5 * 1000 * 1000 * 1000
    BLOCKS = True

    def __init__(self, extractors, chunk_size=DEFAULT_CHUNK_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.model = None
        self.chunk_size = chunk_size

    def __getstate__(self):
        state = super().__getstate__()
        del state["extractors"]
        del state["model"]
        if self.model is not None:
            state["keys"] = self.model.order
        return state

    def __call__(self, processed):
        avglen = processed.values().map(len).mean()
        ndocs = self.extractors[0].ndocs
        self._log.info("Average bag length: %.1f", avglen)
        self._log.info("Number of documents: %d", ndocs)
        self.model = OrderedDocumentFrequencies().construct(
            self.extractors[0].ndocs, [e.docfreq for e in self.extractors])
        chunklen = int(self.chunk_size / (2 * 4 * avglen))
        nparts = ndocs // chunklen + 1
        chunklen = int(ndocs / nparts * (2 * 4 * avglen))
        self._log.info("chunk %d\tparts %d", chunklen, nparts)
        return processed.mapValues(self.bag2row).repartition(nparts)

    def bag2row(self, bag):
        data = numpy.zeros(len(bag), dtype=numpy.float32)
        indices = numpy.zeros(len(bag), dtype=numpy.int32)
        # self.keys emerges after __getstate__
        for i, (k, v) in enumerate(sorted((self.keys[k], v) for k, v in bag.items())):
            data[i] = v
            indices[i] = k
        return data, indices


class BagsBatchSaver(Transformer):
    def __init__(self, path, batcher, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.batcher = batcher
        self.vocabulary_size = 0

    def __getstate__(self):
        state = super().__getstate__()
        del state["batcher"]
        return state

    def __call__(self, head):
        self._log.info("Writing to %s", self.path)
        self.vocabulary_size = len(self.batcher.model)
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
                 "rows": indptr.shape[0] - 1,
                 "cols": self.vocabulary_size}]


BagsBatch = namedtuple("BagsBatch", ("keys", "matrix"))


class BagsBatchParquetLoader(PickleableLogger):
    class BagsBatchParquetLoaderIterator(PickleableLogger):
        def __init__(self, files,**kwargs):
            super().__init__(**kwargs)
            self._files = files
            self._pos = 0

        def _get_log_name(self):
            return BagsBatchParquetLoader.__name__

        def __next__(self):
            if self._pos < len(self._files):
                f = self._files[self._pos]
                self._pos += 1
                try:
                    with open(f, "rb") as fin:
                        rows = list(parquet.DictReader(fin))
                        if len(rows) != 1:
                            raise ValueError("%s contains more than one row" % f)
                        row = rows[0]
                        keys = row["keys"].decode().split("\0")
                        matrix = csr_matrix((numpy.frombuffer(row["data"], numpy.float32),
                                             numpy.frombuffer(row["indices"], numpy.int32),
                                             numpy.frombuffer(row["indptr"], numpy.int64)),
                                            shape=(row["rows"], row["cols"]))
                        return BagsBatch(keys=keys, matrix=matrix)
                except Exception as e:
                    self._log.error("Loading %s: %s: %s", f, type(e).__name__, e)
            raise StopIteration()

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self._files = glob(os.path.join(path, "*.parquet"))

    def _get_log_name(self):
        return type(self).__name__

    def __iter__(self):
        return self.BagsBatchParquetLoaderIterator(self._files, log_level=self._log.level)
