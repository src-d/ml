import logging

from scipy.sparse import csr_matrix

from ast2vec.bow import NBOW
from ast2vec.df import DocumentFrequencies
from ast2vec.id2vec import Id2Vec
from ast2vec.model2.uast2bow import Uasts2BOW
from ast2vec.repo2.base import RepoTransformer, Repo2Base, repos2_entry, repo2_entry
from modelforge.backends import create_backend


class Repo2nBOW(Repo2Base):
    """
    Implements the step repository -> :class:`ast2vec.nbow.NBOW`.
    """
    MODEL_CLASS = NBOW

    def __init__(self, id2vec, docfreq, **kwargs):
        super().__init__(**kwargs)
        self._uasts2bow = Uasts2BOW(id2vec, docfreq, lambda x: x.response.uast)

    def convert_uasts(self, file_uast_generator):
        return self._uasts2bow(file_uast_generator)


class Repo2nBOWTransformer(RepoTransformer):
    """
    Wrap the step: repository -> :class:`ast2vec.nbow.NBOW`.
    """
    WORKER_CLASS = Repo2nBOW

    def __init__(self, id2vec=None, docfreq=None, gcs_bucket=None, **kwargs):
        if gcs_bucket:
            backend = create_backend("gcs", "bucket=" + gcs_bucket)
        else:
            backend = None
        self._id2vec = kwargs["id2vec"] = Id2Vec().load(id2vec or None, backend=backend)
        self._df = kwargs["docfreq"] = DocumentFrequencies().load(docfreq or None, backend=backend)
        prune_df = kwargs.pop("prune_df", 1)
        if prune_df > 1:
            self._df = self._df.prune(prune_df)
        super().__init__(**kwargs)

    def dependencies(self):
        return self._df, self._id2vec

    def result_to_model_kwargs(self, result, url_or_path: str):
        if not result:
            raise ValueError("Empty bag")
        csr_data = []
        csr_indices = []
        csr_indptr = [0, len(result)]
        for key, val in sorted(result.items()):
            csr_data.append(val)
            csr_indices.append(key)
        return {"repos": [url_or_path],
                "matrix": csr_matrix((csr_data, csr_indices, csr_indptr),
                                     shape=(1, len(self._id2vec.tokens)))}


def repo2nbow_entry(args):
    return repo2_entry(args, Repo2nBOWTransformer)


def repos2nbow_entry(args):
    return repos2_entry(args, Repo2nBOWTransformer)
