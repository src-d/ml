from modelforge import Model, assemble_sparse_matrix, disassemble_sparse_matrix, register_model


@register_model
class VocabularyCooccurrences(Model):
    """
    Co-occurrence matrix.
    """
    NAME = "vocabulary_co-occurrences"

    def construct(self, matrix):
        self._matrix = matrix
        return self

    def _load_tree(self, tree):
        self.construct(matrix=assemble_sparse_matrix(tree["matrix"]))

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

    def _generate_tree(self):
        return {"matrix": disassemble_sparse_matrix(self.matrix)}
