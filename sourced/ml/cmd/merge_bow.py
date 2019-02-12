import logging

from sourced.ml.cmd.args import handle_input_arg
from sourced.ml.models import MergeBOW


def merge_bow(args):
    log = logging.getLogger("merge_bow")
    MergeBOW(args.features).convert(handle_input_arg(args.input, log), args.output)
