import logging

from sourced.ml.models import MergeDocFreq
from sourced.ml.cmd.args import handle_input_arg


def merge_df(args):
    log = logging.getLogger("merge_df")
    merger = MergeDocFreq(ordered=args.ordered,
                          vocabulary_size=args.vocabulary_size,
                          min_docfreq=args.min_docfreq)
    merger.convert(handle_input_arg(args.input, log), args.output)
