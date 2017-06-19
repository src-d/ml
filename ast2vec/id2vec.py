from ast2vec.model import Model, split_strings


class Id2Vec(Model):
    NAME = "id2vec"
    DEFAULT_CACHE_DIR = "~/.cache/source{d}/id2vec"

    def _load(self, npz):
        self._embeddings = npz["embeddings"]
        self._tokens = split_strings(npz["tokens"])
        self._log.info("Building the token index...")
        # numpy arrays of string are of fixed item size (max length) - fix this
        self._tokens = list(self._tokens)
        self._token2index = {w: i for i, w in enumerate(self._tokens)}

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def tokens(self):
        return self._tokens

    @property
    def token2index(self):
        return self._token2index


def print_id2vec(tree, dependencies):
    words = split_strings(tree["tokens"])
    embeddings = tree["embeddings"]
    print("Shape:", embeddings.shape)
    print("First 10 words:", words[:10])
