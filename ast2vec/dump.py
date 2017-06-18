import gzip
import numpy

from ast2vec.nbow import print_nbow
from ast2vec.id2vec import print_id2vec


PRINTERS = {
    "nbow": print_nbow,
    "id2vec": print_id2vec,
}


def dump_model(args):
    if args.input.endswith(".gz"):
        with gzip.open(args.input) as f:
            npz = numpy.load(f)
    else:
        npz = numpy.load(args.input)
    meta = npz["meta"]
    if isinstance(meta, numpy.ndarray):
        meta = meta.tolist()
    print(meta)
    try:
        PRINTERS[meta["model"]](npz, args.dependency)
    except KeyError:
        pass
