from modelforge.model import Model, split_strings, assemble_sparse_matrix
from modelforge.models import register_model


@register_model
class Cooccurrences(Model):
    """
    Co-occurrence matrix.
    """
    NAME = "co-occurrences"

    def load(self, tree):
        self._tokens = split_strings(tree["tokens"])
        self._matrix = assemble_sparse_matrix(tree["matrix"])

    def dump(self):
        return """Number of words: %d
First 10 words: %s
Matrix: shape: %s non-zero: %d""" % (
            len(self.tokens), self.tokens[:10], self.matrix.shape, self.matrix.getnnz())

    @property
    def tokens(self):
        """
        Returns the tokens in the order which corresponds to the matrix's rows and cols.
        """
        return self._tokens

    @property
    def matrix(self):
        """
        Returns the sparse co-occurrence matrix.
        """
        return self._matrix

    def __len__(self):
        """
        Returns the number of tokens in the model.
        """
        return len(self._tokens)
