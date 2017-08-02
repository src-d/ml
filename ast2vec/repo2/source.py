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
        uasts = []
        filenames = []

        for file_uast in file_uast_generator:
            sources = self._get_sources(file_uast.filepath)
            src_codes.append(sources)
            uasts.append(file_uast.response)
            filenames.append(file_uast.filename)

        return filenames, src_codes, uasts

    def _get_sources(self, filename):
        try:
            with open(filename, "r", encoding="utf8") as f:
                return f.read()
        except UnicodeDecodeError as e:
            self._log.warning('Skipping file %s.\n\tUnicodeDecodeError: %s', filename, e)


class Repo2SourceTransformer(RepoTransformer):
    WORKER_CLASS = Repo2Source

    def dependencies(self):
        return []

    def result_to_model_kwargs(self, result, url_or_path):
        filenames, src_codes, uasts = result
        return {
            "filenames": filenames,
            "sources": src_codes,
            "uasts": uasts
        }
