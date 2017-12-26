import argparse

from sourced.ml.models import NBOW, BOW


def nbow2bow_entry(args: argparse.Namespace):
    bow = NBOW.as_bow(args.nbow, args.id2vec)
    bow.save(args.output)


def bow2vw_entry(args: argparse.Namespace):
    if not args.nbow:
        bow = BOW().load(source=args.bow)
    else:
        bow = NBOW.as_bow(args.nbow, args.id2vec)
    bow.convert_bow_to_vw(args.output)
