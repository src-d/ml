from ast2vec.model import Model, split_strings


class DocumentFrequencies(Model):
    """
    Document frequencies - number of times a source code identifier appeared
    in different repositories. Each repository counts only once.
    """
    NAME = "docfreq"

    def _load(self, tree):
        self._docs = tree["docs"]
        tokens = split_strings(tree["tokens"])
        freqs = tree["freqs"]
        self._log.info("Building the docfreq dictionary...")
        self._df = dict(zip(tokens, freqs))
        del tokens

    @property
    def docs(self):
        """
        Returns the number of documents.
        """
        return self._docs

    def __getitem__(self, item):
        return self._df[item]

    def get(self, item, default):
        """
        Return the document frequency for a given token.

        :param item: The token to query.
        :param default: Returned value in case the token is missing.
        :return: int
        """
        return self._df.get(item, default)

    def __len__(self):
        """
        Returns the number of tokens in the model.
        """
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
