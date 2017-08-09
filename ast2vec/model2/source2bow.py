from collections import defaultdict
import math

import numpy
from scipy.sparse import csr_matrix

from ast2vec.source import Source
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


class UastModel2BOW(Model2Base):
    MODEL_FROM_CLASS = UASTModel
    MODEL_TO_CLASS = UASTModel

    def __init__(self, topn, docfreq, *args, **kwargs):
        super(UastModel2BOW, self).__init__(*args, **kwargs)
        self._log.info("Choosing the vocabulary...")
        freqs = numpy.zeros(len(docfreq), dtype=int)
        vocabulary = [None] * len(docfreq)
        for i, (k, v) in enumerate(docfreq):
            freqs[i] = -v
            vocabulary[i] = k
        indices = freqs.argpartition(topn)[:topn]
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
        bow = BOW()
        bow.construct(repos=[model.repository], matrix=matrix, tokens=self._tokens)
        bow.meta["dependencies"] = [self._uasts2bow.docfreq]
        return bow


class Source2BOW(UastModel2BOW):
    MODEL_FROM_CLASS = Source
    MODEL_TO_CLASS = Source


def source2bow_entry(args):
    df = DocumentFrequencies().load(args.df)
    converter = Source2BOW(args.vocabulary_size, df, num_processes=args.processes)
    converter.convert(args.input, args.output, pattern=args.filter)
