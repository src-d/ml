from modelforge.model import Model, split_strings, assemble_sparse_matrix
from modelforge.models import register_model


@register_model
class NBOW(Model):
    """
    Weighted bag of words model. Every word is represented with a vector.
    Bag = repository. Word = source code identifier. This model depends on
    :class:`ast2vec.id2vec.Id2Vec` and :class:`ast2vec.df.DocumentFrequencies`.
    """

    NAME = "nbow"

    def load(self, tree):
        self._repos = split_strings(tree["repos"])
        self._matrix = assemble_sparse_matrix(tree["matrix"])
        self._log.info("Building the repository names mapping...")
        self._repos_map = {r: i for i, r in enumerate(self._repos)}

    def dump(self):
        return """Shape: %s
First 10 repos: %s""" % (
            self._matrix.shape, self._repos[:10])

    def __getitem__(self, item):
        """
        Returns repository name, word indices and weights for the given
        repository index.

        :param item: Repository index.
        :return: (name, :class:`numpy.ndarray` with word indices, \
                  :class:`numpy.ndarray` with weights)
        """
        data = self._matrix[item]
        return self._repos[item], data.indices, data.data

    def __iter__(self):
        """
        Returns an iterator over the repository indices.
        """
        return iter(range(len(self)))

    def __len__(self):
        """
        Returns the number of repositories.
        """
        return len(self._repos)

    def repository_index_by_name(self, name):
        """
        Looks up repository index by it's name.
        """
        return self._repos_map[name]
