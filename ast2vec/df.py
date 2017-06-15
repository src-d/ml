from ast2vec.dataset import Dataset


class DocumentFrequencies(Dataset):
    LOG_NAME = "df"
    DEFAULT_CACHE_DIR = "~/.cache/source{d}/df"

    def _load(self, npz):
        tokens = npz["tokens"]
        freqs = npz["freqs"]
        self._df = dict(zip(tokens, freqs))
        self._sum = freqs.sum()

    @property
    def sum(self):
        return self._sum

    def __getitem__(self, item):
        return self._df[item]

    def get(self, item, default):
        return self._df.get(item, default)

    def __len__(self):
        return len(self._df)
