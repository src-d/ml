from modelforge.model import Model, split_strings
from modelforge.models import register_model


@register_model
class Id2Vec(Model):
    """
    id2vec model - source code identifier embeddings.
    """
    NAME = "id2vec"

    def load(self, tree):
        self._embeddings = tree["embeddings"].copy()
        self._tokens = split_strings(tree["tokens"])
        self._log.info("Building the token index...")
        self._token2index = {w: i for i, w in enumerate(self._tokens)}

    def dump(self):
        return """Shape: %s
First 10 words: %s""" % (
            self.embeddings.shape, self.tokens[:10])

    @property
    def embeddings(self):
        """
        :class:`numpy.ndarray` with the embeddings of shape
        (N tokens x embedding dims).
        """
        return self._embeddings

    @property
    def tokens(self):
        """
        List with the processed source code identifiers.
        """
        return self._tokens

    def __getitem__(self, item):
        """
        Returns the index of the specified processed source code identifier.
        """
        return self._token2index[item]
