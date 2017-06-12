import os

import numpy


def parse_swivel_embeddings(directory):
    tokens = []
    embeddings = []
    with open(os.path.join(directory, "row_embedding.tsv")) as frow:
        with open(os.path.join(directory, "col_embedding.tsv")) as fcol:
            for lrow, lcol in zip(frow, fcol):
                prow, pcol = (l.split("\t", 1) for l in (lrow, lcol))
                assert prow[0] == pcol[0]
                tokens.append(prow[0])
                erow, ecol = (numpy.fromstring(p[1], dtype=numpy.float32,
                                               sep="\t")
                              for p in (prow, pcol))
                embeddings.append((erow + ecol) / 2)
    return numpy.array(embeddings), tokens
