from modelforge.model import Model, split_strings
from modelforge.models import register_model


@register_model
class DocumentFrequencies(Model):
    """
    Document frequencies - number of times a source code identifier appeared
    in different repositories. Each repository counts only once.
    """
    NAME = "docfreq"

    def load(self, tree):
        self._docs = tree["docs"]
        tokens = split_strings(tree["tokens"])
        freqs = tree["freqs"]
        self._log.info("Building the docfreq dictionary...")
        self._df = dict(zip(tokens, freqs))
        del tokens

    def dump(self):
        return """Number of words: %d
First 10 words: %s
Number of documents: %d""" % (
            len(self._df), self.tokens()[:10], self.docs)

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

    def tokens(self):
        """
        Returns the sorted list of tokens.
        """
        return sorted(self._df)

    def __len__(self):
        """
        Returns the number of tokens in the model.
        """
        return len(self._df)
