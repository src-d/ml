from collections import defaultdict
import logging
import math
import os

import asdf

from ast2vec.meta import generate_meta, ARRAY_COMPRESSION
from ast2vec.id2vec import Id2Vec
from ast2vec.df import DocumentFrequencies
from ast2vec.nbow import NBOW
from ast2vec.repo2base import Repo2Base, RepoTransformer, repos2_entry, \
    ensure_bblfsh_is_running_noexc, repo2_entry


class Repo2nBOW(Repo2Base):
    """
    Implements the step repository -> :class:`ast2vec.nbow.NBOW`.
    """
    MODEL_CLASS = NBOW

    def __init__(self, id2vec, docfreq, tempdir=None, linguist=None,
                 log_level=logging.INFO, bblfsh_endpoint=None,
                 timeout=Repo2Base.DEFAULT_BBLFSH_TIMEOUT):
        super(Repo2nBOW, self).__init__(
            tempdir=tempdir, linguist=linguist, log_level=log_level,
            bblfsh_endpoint=bblfsh_endpoint, timeout=timeout)
        self._id2vec = id2vec
        self._docfreq = docfreq

    @property
    def id2vec(self):
        return self._id2vec

    @property
    def docfreq(self):
        return self._docfreq

    def convert_uasts(self, uast_generator):
        freqs = defaultdict(int)
        for uast in uast_generator:
            bag = self._uast_to_bag(uast.uast)
            for key, freq in bag.items():
                freqs[key] += freq
        missing = []
        vocabulary = self._id2vec.tokens
        for key, val in freqs.items():
            try:
                freqs[key] = math.log(1 + val) * math.log(
                    self._docfreq.docs / self._docfreq[vocabulary[key]])
            except KeyError:
                missing.append(key)
        for key in missing:
            del freqs[key]
        return dict(freqs)

    def _uast_to_bag(self, uast):
        stack = [uast]
        bag = defaultdict(int)
        while stack:
            node = stack.pop(0)
            if self.SIMPLE_IDENTIFIER in node.roles:
                for sub in self._process_token(node.token):
                    try:
                        bag[self._id2vec[sub]] += 1
                    except KeyError:
                        pass
            stack.extend(node.children)
        return bag


class Repo2nBOWTransformer(RepoTransformer):
    """
    Wrap the step: repository -> :class:`ast2vec.nbow.NBOW`.
    """
    WORKER_CLASS = Repo2nBOW

    def __init__(self, id2vec=None, docfreq=None, gcs_bucket=None, **kwargs):
        self._id2vec = kwargs["id2vec"] = Id2Vec(id2vec or None, gcs_bucket=gcs_bucket)
        self._df = kwargs["docfreq"] = DocumentFrequencies(docfreq or None, gcs_bucket=gcs_bucket)
        super(Repo2nBOWTransformer, self).__init__(**kwargs)

    def result_to_tree(self, result):
        if not result:
            raise ValueError("Empty bag")
        return {
            "nbow": result,
            "meta": generate_meta(self.WORKER_CLASS.MODEL_CLASS.NAME, self._id2vec, self._df)
        }


def repo2nbow_entry(args):
    return repo2_entry(args, Repo2nBOWTransformer)


def repos2nbow_entry(args):
    return repos2_entry(args, Repo2nBOWTransformer)
