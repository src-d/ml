from ast2vec.model import Model, split_strings


class DocumentFrequencies(Model):
    """
    Document frequencies - number of times a source code identifier appeared
    in different repositories. Each repository counts only once.
    """
    NAME = "docfreq"

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
    """
    Prints the brief information about :class:`DocumentFrequencies` model.
    :param tree: Internal loaded tree of the model.
    :param dependencies: Not used.
    :return: None
    """
    words = split_strings(tree["tokens"])
    print("Number of words:", len(words))
    print("First 10 words:", words[:10])
