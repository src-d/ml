from pprint import pprint

from ast2vec.df import print_df
from ast2vec.model import Model
from ast2vec.nbow import print_nbow
from ast2vec.id2vec import print_id2vec
from ast2vec.repo2coocc import print_coocc


PRINTERS = {
    "nbow": print_nbow,
    "id2vec": print_id2vec,
    "docfreq": print_df,
    "co-occurrences": print_coocc
}


class GenericModel(Model):
    def _load(self, tree):
        self.tree = tree


def dump_model(args):
    model = GenericModel(args.input, gcs_bucket=args.gcs)
    meta = model.meta
    pprint(meta)
    try:
        PRINTERS[meta["model"]](model.tree, args.dependency)
    except KeyError:
        pass
