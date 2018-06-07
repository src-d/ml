import argparse

from sourced.ml.models import BOW


def bow2vw_entry(args: argparse.Namespace):
    bow = BOW().load(source=args.bow)
    bow.convert_bow_to_vw(args.output)
