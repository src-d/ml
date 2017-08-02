from bblfsh.github.com.bblfsh.sdk.protocol.generated_pb2 import ParseResponse
from modelforge import generate_meta
from modelforge.model import Model, split_strings, merge_strings, write_model
from modelforge.models import register_model

import ast2vec


@register_model
class Source(Model):
    """
    Model for source-code storage
    """
    NAME = "source"

    def construct(self, filenames, sources, uasts):
        self._filenames = filenames
        self._sources = sources
        self._uasts = uasts

    def _load_tree(self, tree):
        self.construct(filenames=split_strings(tree["filenames"]),
                       sources=split_strings(tree["sources"]),
                       uasts=[ParseResponse.FromString(x) for x in tree["uasts"]])

    def dump(self):
        symbols_num = 100
        out = self._sources[0][:symbols_num]
        return "Number of files: %d. First %d symbols:\n %s" % (
            len(self._filenames), symbols_num, out)

    @property
    def sources(self):
        """
        Returns all code files of the saved repo
        """
        return self._sources

    @property
    def uasts(self):
        """
        Returns all usts of code in the saved repo
        """
        return self._uasts

    @property
    def filenames(self):
        """
        Returns all filenames in the saved repo
        """
        return self._filenames

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, ast2vec.__version__, *deps)
        write_model(self._meta,
                    {"filenames": merge_strings(self.filenames),
                     "sources": merge_strings(self.sources),
                     "uasts": [uast.SerializeToString() for uast in self.uasts]},
                    output)
