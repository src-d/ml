from ast2vec.model import Model, split_strings, assemble_sparse_matrix
from ast2vec.id2vec import Id2Vec


class NBOW(Model):
    NAME = "nbow"

    def _load(self, tree):
        self._repos = split_strings(tree["repos"])
        self._model = assemble_sparse_matrix(tree["matrix"])
        self._log.info("Building the repository names mapping...")
        self._repos_map = {r: i for i, r in enumerate(self._repos)}

    def __getitem__(self, item):
        data = self._model[item]
        return self._repos[item], data.indices, data.data

    def __iter__(self):
        return iter(range(len(self)))

    def __len__(self):
        return len(self._repos)

    def repository_index_by_name(self, name):
        return self._repos_map[name]


def print_nbow(tree, dependencies):
    try:
        nbow = tree["nbow"]
        if dependencies is None:
            dependencies = [d["uuid"] for d in tree["model"]["dependencies"]
                            if d["model"] == "id2vec"]
        id2vec = Id2Vec(source=dependencies[0])
        nbl = [(f, id2vec.tokens[t]) for t, f in nbow.items()]
        nbl.sort(reverse=True)
        for w, t in nbl:
            print("%s\t%f" % (t, w))
    except KeyError:
        print("Shape:", tree["matrix"]["shape"])
        print("First 10 repos:", split_strings(tree["repos"])[:10])
