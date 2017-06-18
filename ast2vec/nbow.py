from ast2vec.dataset import Dataset
from ast2vec.id2vec import Id2Vec


class NBOW(Dataset):
    LOG_NAME = "nbow"
    DEFAULT_CACHE_DIR = "~/.cache/source{d}/nbow"

    def _load(self, npz):
        self._repos = npz["repos"]
        self._model = npz["matrix"]
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


def print_nbow(npz, dependencies):
    nbow = npz["nbow"]
    id2vec = Id2Vec(dependencies[0])
    nbl = [(f, id2vec.tokens[t]) for t, f in nbow.items()]
    nbl.sort(reverse=True)
    for w, t in nbl:
        print("%s\t%f" % (t, w))
