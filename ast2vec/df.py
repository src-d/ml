from ast2vec.model import Model, split_strings


class DocumentFrequencies(Model):
    NAME = "docfreq"
    DEFAULT_CACHE_DIR = "~/.cache/source{d}/df"

    def _load(self, tree):
        tokens = split_strings(tree["tokens"])
        freqs = tree["freqs"]
        self._log.info("Building the docfreq dictionary...")
        self._df = dict(zip(tokens, freqs))
        del tokens
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


def print_df(tree, dependencies):
    words = split_strings(tree["tokens"])
    print("Number of words:", len(words))
    print("First 10 words:", words[:10])
