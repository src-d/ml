from itertools import islice
from ast2vec.model import Model, split_strings, assemble_sparse_matrix
from ast2vec.id2vec import Id2Vec


class NBOW(Model):
    """
    Weighted bag of words model. Every word is represented with a vector.
    Bag = repository. Word = source code identifier. This model depends on
    :class:`ast2vec.id2vec.Id2Vec` and :class:`ast2vec.df.DocumentFrequencies`.
    """

    NAME = "nbow"

    def _load(self, tree):
        self._repos = split_strings(tree["repos"])
        self._model = assemble_sparse_matrix(tree["matrix"])
        self._log.info("Building the repository names mapping...")
        self._repos_map = {r: i for i, r in enumerate(self._repos)}

    def __getitem__(self, item):
        """
        Returns repository name, word indices and weights for the given
        repository index.

        :param item: Repository index.
        :return: (name, :class:`numpy.ndarray` with word indices, \
                  :class:`numpy.ndarray` with weights)
        """
        data = self._model[item]
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


def print_nbow(tree, dependencies):
    """
    Print the brief information about the :class:`NBOW` model.

    :param tree: Internal loaded tree of the model.
    :param dependencies: Overriding parent model sources.
    :return: None
    """
    MAX_WORDS = 10
    try:
        nbow = tree["nbow"]
        if dependencies is None:
            dependencies = [d["uuid"] for d in tree["model"]["dependencies"]
                            if d["model"] == "id2vec"]
        id2vec = Id2Vec(source=dependencies[0])
        nbl = [(f, id2vec.tokens[t]) for t, f in nbow.items()]
        nbl.sort(reverse=True)
        for w, t in islice(nbl, MAX_WORDS):
            print("%s\t%f" % (t, w))
    except KeyError:
        print("Shape:", tree["matrix"]["shape"])
        print("First 10 repos:", split_strings(tree["repos"])[:10])
