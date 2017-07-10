import logging
from pprint import pprint

from ast2vec.df import DocumentFrequencies, print_df
from ast2vec.model import Model
from ast2vec.nbow import NBOW, print_nbow
from ast2vec.id2vec import Id2Vec, print_id2vec
from ast2vec.coocc import Cooccurrences, print_coocc


PRINTERS = {
    NBOW.NAME: print_nbow,
    Id2Vec.NAME: print_id2vec,
    DocumentFrequencies.NAME: print_df,
    Cooccurrences.NAME: print_coocc
}


class GenericModel(Model):
    """
    Automatically loads any model and saves it's internal loaded tree.
    """
    def _load(self, tree):
        self.tree = tree
        self._load_tree(tree)

    def _load_tree(self, tree):
        try:
            tree.__getitem__(0)
        except:
            pass
        if isinstance(tree, dict):
            for val in tree.values():
                self._load_tree(val)
        elif isinstance(tree, list):
            for val in tree:
                self._load_tree(val)


def dump_model(args):
    """
    Prints the information about the model.

    :param args: :class:`argparse.Namespace` with "input", "gcs" and \
                 "dependency". "dependency" overrides the parent models.
    :return: None
    """
    model = GenericModel(args.input, gcs_bucket=args.gcs,
                         log_level=logging._nameToLevel[args.log_level])
    meta = model.meta
    pprint(meta)
    try:
        PRINTERS[meta["model"]](model.tree, args.dependency)
    except KeyError:
        print("Printer for", meta["model"], "couldn\'t parse", args.input)
