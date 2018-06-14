import sys

import numpy
from scipy.sparse import csr_matrix

from sourced.ml.models import Topics


def bigartm2asdf(args):
    """
    BigARTM "readable" model -> Topics -> Modelforge ASDF.
    """
    tokens = []
    data = []
    indices = []
    indptr = [0]
    if args.input != "-":
        fin = open(args.input)
    else:
        fin = sys.stdin
    try:
        # the first line is the header
        fin.readline()
        for line in fin:
            items = line.split(";")
            tokens.append(items[0])
            nnz = 0
            for i, v in enumerate(items[2:]):
                if v == "0":
                    continue
                nnz += 1
                data.append(float(v))
                indices.append(i)
            indptr.append(indptr[-1] + nnz)
    finally:
        if args.input != "-":
            fin.close()
    data = numpy.array(data, dtype=numpy.float32)
    indices = numpy.array(indices, dtype=numpy.int32)
    matrix = csr_matrix((data, indices, indptr), shape=(len(tokens), len(items) - 2)).T
    Topics().construct(tokens, None, matrix).save(args.output)
