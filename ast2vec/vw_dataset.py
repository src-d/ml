import argparse
import logging
from typing import List, Mapping, Union

from modelforge.progress_bar import progress_bar

from ast2vec.id2vec import Id2Vec
from ast2vec.nbow import NBOW


def convert_nbow_to_vw(nbow: NBOW, vocabulary: Union[Mapping[int, str], List[str]], output: str):
    log = logging.getLogger("nbow2vw")
    with open(output, "w") as fout:
        for index in progress_bar(nbow, log, expected_size=len(nbow)):
            record = nbow[index]
            fout.write(record[0] + " ")
            pairs = []
            for t, v in zip(*record[1:]):
                try:
                    pairs.append("%s:%s" % (vocabulary[t], v))
                except (KeyError, IndexError):
                    log.warning("%d not found in the vocabulary", t)
            fout.write(" ".join(pairs))
            fout.write("\n")


def nbow2vw_entry(args: argparse.Namespace):
    nbow = NBOW(source=args.nbow)
    if args.id2vec:
        id2vec = Id2Vec(source=args.id2vec)
    else:
        id2vec = Id2Vec(source=nbow.get_dependency("id2vec")["uuid"])
    convert_nbow_to_vw(nbow, id2vec.tokens, args.output)
