from bblfsh.github.com.bblfsh.sdk.protocol.generated_pb2 import ParseResponse
from modelforge.model import Model, split_strings
from modelforge.models import register_model


@register_model
class Source(Model):
    """
    Model for source-code storage
    """
    NAME = "source"

    def load(self, tree):
        self._filenames = split_strings(tree["filenames"])
        self._sources = split_strings(tree["sources"])
        self._uasts = [ParseResponse.FromString(x) for x in tree["uasts"]]

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
