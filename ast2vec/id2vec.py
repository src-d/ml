from ast2vec.model import Model, split_strings


class Id2Vec(Model):
    """
    id2vec model - source code identifier embeddings.
    """
    NAME = "id2vec"

    def _load(self, tree):
        self._embeddings = tree["embeddings"].copy()
        self._tokens = split_strings(tree["tokens"])
        self._log.info("Building the token index...")
        # numpy arrays of string are of fixed item size (max length) - fix this
        self._tokens = list(self._tokens)
        self._token2index = {w: i for i, w in enumerate(self._tokens)}

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


def print_id2vec(tree, dependencies):
    """
    Prints the brief information about :class:`Id2Vec` model.

    :param tree: Internal loaded tree of the model.
    :param dependencies: Not used.
    :return: None
    """
    words = split_strings(tree["tokens"])
    embeddings = tree["embeddings"]
    print("Shape:", embeddings.shape)
    print("First 10 words:", words[:10])
