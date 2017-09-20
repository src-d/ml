from collections import defaultdict
import logging
import marshal
import math
import os
import pickle
import types

import numpy
from scipy.sparse import csr_matrix

from ast2vec.uast import UASTModel
from ast2vec.df import DocumentFrequencies
from ast2vec.bow import BOW
from ast2vec.model2.base import Model2Base
from ast2vec.uast_ids_to_bag import UastIds2Bag


class Uasts2BOW:
    def __init__(self, vocabulary: dict, docfreq: DocumentFrequencies,
                 getter: callable):
        self._docfreq = docfreq
        self._uast2bag = UastIds2Bag(vocabulary)
        self._reverse_vocabulary = [None] * len(vocabulary)
        for key, val in vocabulary.items():
            self._reverse_vocabulary[val] = key
        self._getter = getter

    @property
    def vocabulary(self):
        return self._uast2bag.vocabulary

    @property
    def docfreq(self):
        return self._docfreq

    def __call__(self, file_uast_generator):
        freqs = defaultdict(int)
        for file_uast in file_uast_generator:
            bag = self._uast2bag.uast_to_bag(self._getter(file_uast))
            for key, freq in bag.items():
                freqs[key] += freq
        missing = []
        for key, val in freqs.items():
            try:
                freqs[key] = math.log(1 + val) * math.log(
                    self._docfreq.docs / self._docfreq[self._reverse_vocabulary[key]])
            except KeyError:
                missing.append(key)
        for key in missing:
            del freqs[key]
        return dict(freqs)

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self._getter, types.FunctionType) \
                and self._getter.__name__ == (lambda: None).__name__:
            assert self._getter.__closure__ is None
            state["_getter"] = marshal.dumps(self._getter.__code__)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if isinstance(self._getter, bytes):
            self._getter = types.FunctionType(marshal.loads(self._getter), globals())


class UastModel2BOW(Model2Base):
    MODEL_FROM_CLASS = UASTModel
    MODEL_TO_CLASS = BOW

    def __init__(self, topn, docfreq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log.info("Choosing the vocabulary...")
        freqs = numpy.zeros(len(docfreq), dtype=int)
        vocabulary = [None] * len(docfreq)
        for i, (k, v) in enumerate(docfreq):
            freqs[i] = -v
            vocabulary[i] = k
        topn = min(topn, len(freqs))
        indices = freqs.argpartition(topn-1)[:topn]
        freqs = freqs[indices]
        vocabulary = [vocabulary[i] for i in indices]
        indices = freqs.argsort()
        self._tokens = [vocabulary[i] for i in indices]
        vocabulary = {t: i for i, t in enumerate(self._tokens)}
        self._uasts2bow = Uasts2BOW(vocabulary, docfreq, lambda x: x)

    def convert_model(self, model: UASTModel) -> BOW:
        bag = self._uasts2bow(model.uasts)
        data = numpy.array(list(bag.values()), dtype=numpy.float32)
        indices = numpy.array(list(bag.keys()), dtype=numpy.int32)
        matrix = csr_matrix((data, indices, [0, len(data)]),
                            shape=(1, len(self._uasts2bow.vocabulary)))
        bow = BOW(log_level=logging.WARNING)
        bow.construct(repos=[model.repository], matrix=matrix, tokens=self._tokens)
        bow.meta["dependencies"] = [self._uasts2bow.docfreq]
        return bow


def uast2bow_entry(args):
    df = DocumentFrequencies().load(args.docfreq)
    if args.prune_df > 1:
        df = df.prune(args.prune_df)
    os.makedirs(args.output, exist_ok=True)
    converter = UastModel2BOW(args.vocabulary_size, df, num_processes=args.processes,
                              overwrite_existing=args.overwrite_existing)
    converter.convert(args.input, args.output, pattern=args.filter)
