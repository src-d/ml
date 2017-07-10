from ast2vec.model import Model, split_strings, assemble_sparse_matrix


class Cooccurrences(Model):
    """
    Co-occurrence matrix.
    """
    NAME = "co-occurrences"

    def _load(self, tree):
        self._tokens = split_strings(tree["tokens"])
        self._matrix = assemble_sparse_matrix(tree["matrix"])

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


def print_coocc(tree, dependencies):
    """
    Prints the brief information about :class:`Cooccurrences` model.

    :param tree: Internal loaded tree of the model.
    :param dependencies: Not used.
    :return: None
    """

    words = split_strings(tree["tokens"])
    m_shape = tree["matrix"]["shape"]
    nnz = tree["matrix"]["data"][0].shape[0]

    print("Number of words:", len(words))
    print("First 10 words:", words[:10])
    print("Matrix:", ", shape:", m_shape, "number of non zero elements", nnz)
