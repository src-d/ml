from ast2vec.dataset import Dataset


class Id2Vec(Dataset):
    LOG_NAME = "id2vec"
    DEFAULT_CACHE_DIR = "~/.cache/source{d}/id2vec"

    def _load(self, npz):
        self._embeddings = npz["embeddings"]
        self._tokens = npz["tokens"]
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
