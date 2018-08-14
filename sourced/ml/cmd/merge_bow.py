import logging

from sourced.ml.models import MergeBOW
from sourced.ml.cmd.args import handle_input_arg


def merge_bow(args):
    log = logging.getLogger("merge_bow")
    MergeBOW(args.features).convert(handle_input_arg(args.input, log), args.output)
