from modelforge.model import Model, assemble_sparse_matrix
from modelforge.models import register_model


@register_model
class VocabularyCooccurrences(Model):
    """
    Co-occurrence matrix.
    """
    NAME = "vocabulary_co-occurrences"

    def load(self, tree):
        self._matrix = assemble_sparse_matrix(tree["matrix"])

    def dump(self):
        return """
Matrix: shape: %s non-zero: %d""" % (
            self.matrix.shape, self.matrix.getnnz())

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
        return self._matrix.shape[0]
