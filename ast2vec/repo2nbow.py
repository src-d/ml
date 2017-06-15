from collections import defaultdict
import logging
import math

from ast2vec.id2vec import Id2Vec
from ast2vec.df import DocumentFrequencies
from ast2vec.repo2base import Repo2Base


class Repo2nBOW(Repo2Base):
    LOG_NAME = "repo2nbow"

    def __init__(self, id2vec, docfreq, tempdir=None, linguist=None,
                 log_level=logging.INFO, bblfsh_endpoint=None):
        super(Repo2nBOW, self).__init__(
            tempdir=tempdir, linguist=linguist, log_level=log_level,
            bblfsh_endpoint=bblfsh_endpoint)
        self._id2vec = id2vec
        self._docfreq = docfreq

    def convert_uasts(self, uast_generator):
        freqs = defaultdict(int)
        for uast in uast_generator:
            bag = self._uast_to_bag(uast)
            for key, freq in bag.items():
                freqs[key] += freq
        for key, val in freqs:
            freqs[key] = math.log(1 + val) * math.log(
                self._docfreq.sum / self._docfreq[key])
        return freqs

    def _uast_to_bag(self, uast):
        stack = [uast]
        bag = defaultdict(int)
        while stack:
            node = stack.pop(0)
            if self.SIMPLE_IDENTIFIER in node.roles:
                for sub in self._process_token(node.token):
                    bag[sub] += 1
            stack.extend(node.children)
        return bag


def repo2nbow(url_or_path, id2vec=None, df=None, linguist=None):
    if id2vec is None:
        id2vec = Id2Vec()
    if df is None:
        df = DocumentFrequencies()
    obj = Repo2nBOW(id2vec, df, linguist=linguist)
    nbow = obj.convert_repository(url_or_path)
    return nbow


def repo2nbow2stdout(args):
    id2vec = Id2Vec(args.id2vec or None)
    df = DocumentFrequencies(args.df or None)
    linguist = args.linguist or None
    nbow = repo2nbow(args.repository, id2vec=id2vec, df=df, linguist=linguist)
    nbl = [(weight, token) for token, weight in nbow.items()]
    nbl.sort(reverse=True)
    for w, t in nbl:
        print("%s\t%f\n" % (t, w))
