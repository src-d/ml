from modelforge.meta import generate_meta
from modelforge.model import merge_strings

import ast2vec
from ast2vec.repo2.base import Repo2Base
from ast2vec.repo2.base import RepoTransformer
from ast2vec.source import Source


class Repo2Source(Repo2Base):
    """
    Extract source code and uast of repository of certain languages
    """
    MODEL_CLASS = Source

    def __init__(self, *args, **kwargs):
        super(Repo2Source, self).__init__(*args, **kwargs)
        self._uast_only = False

    def convert_uasts(self, file_uast_generator):
        src_codes = []
        uast_protos = []
        filenames = []

        for file_uast in file_uast_generator:
            src_codes.append(self._get_sources(file_uast.filepath))
            uast_protos.append(self._uast_to_proto(file_uast.response))
            filenames.append(file_uast.filename)

        return src_codes, uast_protos, filenames

    def _get_sources(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            return f.read()

    def _uast_to_proto(self, uast):
        return uast.SerializeToString()


class Repo2SourceTransformer(RepoTransformer):
    WORKER_CLASS = Repo2Source

    @classmethod
    def result_to_tree(cls, result):
        src_codes, uast_protos, filenames = result
        return {
            "filenames": merge_strings(filenames),
            "sources": merge_strings(src_codes),
            "uasts": uast_protos,
            "meta": generate_meta(cls.WORKER_CLASS.MODEL_CLASS.NAME, ast2vec.__version__)
        }
