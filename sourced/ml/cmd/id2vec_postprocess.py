import logging
import os
import sys

import numpy

from sourced.ml.algorithms import TokenParser
from sourced.ml.models import Id2Vec


def id2vec_postprocess(args):
    """
    Merges row and column embeddings produced by Swivel and writes the Id2Vec
    model.

    :param args: :class:`argparse.Namespace` with "swivel_data" \
                 and "result". The text files are read from \
                 `swivel_data` and the model is written to \
                 `result`.
    :return: None
    """
    log = logging.getLogger("postproc")
    log.info("Parsing the embeddings at %s...", args.swivel_data)
    tokens = []
    embeddings = []
    swd = args.swivel_data
    with open(os.path.join(swd, "row_embedding.tsv")) as frow:
        with open(os.path.join(swd, "col_embedding.tsv")) as fcol:
            for i, (lrow, lcol) in enumerate(zip(frow, fcol)):
                if i % 10000 == (10000 - 1):
                    sys.stdout.write("%d\r" % (i + 1))
                    sys.stdout.flush()
                prow, pcol = (l.split("\t", 1) for l in (lrow, lcol))
                assert prow[0] == pcol[0]
                tokens.append(prow[0][:TokenParser.MAX_TOKEN_LENGTH])
                erow, ecol = \
                    (numpy.fromstring(p[1], dtype=numpy.float32, sep="\t")
                     for p in (prow, pcol))
                embeddings.append((erow + ecol) / 2)
    log.info("Generating numpy arrays...")
    embeddings = numpy.array(embeddings, dtype=numpy.float32)
    log.info("Writing %s...", args.output)
    Id2Vec().construct(embeddings=embeddings, tokens=tokens).save(args.output)
