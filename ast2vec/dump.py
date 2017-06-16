import numpy

from ast2vec.repo2nbow import print_nbow


PRINTERS = {
    "nbow": print_nbow
}


def dump_model(args):
    npz = numpy.load(args.input)
    print(npz["meta"])
    try:
        PRINTERS[npz["model"]](npz, args.dependency)
    except KeyError:
        pass
