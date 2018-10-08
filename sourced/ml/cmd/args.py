import argparse
import json
import logging
from typing import Optional, Union, Iterable
import sys


from sourced.ml import extractors
from sourced.ml.transformers import BOWWriter, Moder
from sourced.ml.utils import add_engine_args


class ArgumentDefaultsHelpFormatterNoNone(argparse.ArgumentDefaultsHelpFormatter):
    """
    Pretty formatter of help message for arguments.
    It adds default value to the end if it is not None.
    """
    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def handle_input_arg(input_arg: Union[str, Iterable[str]],
                     log: Optional[logging.Logger] = None):
    """
    Process input arguments and return an iterator over input files.

    :param input_arg: list of files to process or `-` to get \
        file paths from stdin.
    :param log: Logger if you want to log handling process.
    :return: An iterator over input files.
    """
    log = log.info if log else (lambda *x: None)
    if input_arg == "-" or input_arg == ['-']:
        log("Reading file paths from stdin.")
        for line in sys.stdin:
            yield line.strip()
    else:
        if isinstance(input_arg, str):
            yield input_arg
        else:
            yield from input_arg


def add_repartitioner_arg(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "--partitions", required=False, default=None, type=int,
        help="Performs data repartition to specified number of partitions. "
             "Nothing happens if parameter is unset.")
    my_parser.add_argument(
        "--shuffle", action="store_true",
        help="Use RDD.repartition() instead of RDD.coalesce().")


def add_split_stem_arg(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "--split", action="store_true",
        help="Split identifiers based on special characters or case changes. For example split "
             "'ThisIs_token' to ['this', 'is', 'token'].")


def add_vocabulary_size_arg(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "-v", "--vocabulary-size", default=10000000, type=int,
        help="The maximum vocabulary size.")


def add_min_docfreq(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "--min-docfreq", default=1, type=int,
        help="The minimum document frequency of each feature.")


def add_repo2_args(my_parser: argparse.ArgumentParser, default_packages=None):
    my_parser.add_argument(
        "-r", "--repositories", required=True,
        help="The path to the repositories.")
    my_parser.add_argument(
        "--parquet", action="store_true", help="Use Parquet files as input.")
    my_parser.add_argument(
        "--graph", help="Write the tree in Graphviz format to this file.")
    # TODO(zurk): get languages from bblfsh directly as soon as
    # https://github.com/bblfsh/client-scala/issues/98 resolved
    languages = ["Java", "Python", "Go", "JavaScript", "TypeScript", "Ruby", "Bash", "Php"]
    my_parser.add_argument(
        "-l", "--languages", nargs="+", choices=languages,
        default=None,  # Default value for --languages arg should be None.
                       # Otherwise if you process parquet files without 'lang' column, you will
                       # fail to process it with any --languages argument.
        help="The programming languages to analyse.")
    my_parser.add_argument("--blacklist", action="store_true",
                           help="Exclude the languages in --languages from the analysis "
                                "instead of filtering by default.")
    add_dzhigurda_arg(my_parser)
    add_engine_args(my_parser, default_packages)


def add_df_args(my_parser: argparse.ArgumentParser, required=True):
    my_parser.add_argument(
        "--min-docfreq", default=1, type=int,
        help="The minimum document frequency of each feature.")
    df_group = my_parser.add_mutually_exclusive_group(required=required)
    df_group.add_argument(
        "--docfreq-out", help="Path to save generated DocumentFrequencies model.")
    df_group.add_argument(
        "--docfreq-in", help="Path to load pre-generated DocumentFrequencies model.")
    add_vocabulary_size_arg(my_parser)


def add_feature_args(my_parser: argparse.ArgumentParser, required=True):
    my_parser.add_argument("-x", "--mode", choices=Moder.Options.__all__,
                           default="file", help="What to select for analysis.")
    my_parser.add_argument(
        "--quant", help="[IN/OUT] The path to the QuantizationLevels model.")
    my_parser.add_argument(
        "-f", "--feature", nargs="+",
        choices=[ex.NAME for ex in extractors.__extractors__.values()],
        required=required, help="The feature extraction scheme to apply.")
    for ex in extractors.__extractors__.values():
        for opt, val in ex.OPTS.items():
            my_parser.add_argument(
                "--%s-%s" % (ex.NAME, opt), default=val, type=json.loads,
                help="%s's kwarg" % ex.__name__)


def add_bow_args(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "--bow", required=True, help="[OUT] The path to the Bag-Of-Words model.")
    my_parser.add_argument(
        "--batch", default=BOWWriter.DEFAULT_CHUNK_SIZE, type=int,
        help="The maximum size of a single BOW file in bytes.")
    my_parser.add_argument(
        "--num-iterations", default=1, type=int,
        help="After partitioning by document we run the pipeline on each partition separately "
             "in a loop. This number indicates the number of partitions.")


def add_cached_index_arg(my_parser: argparse.ArgumentParser, create: bool = False):
    direction = "OUT" if create else "IN"
    my_parser.add_argument(
        "--cached-index-path", default=None, required=True,
        help="[%s] Path to the docfreq model holding the document's index." % direction)


def add_dzhigurda_arg(my_parser):
    my_parser.add_argument(
        "--dzhigurda", default=0, type=int,
        help="Number of the additional commits look over in the history starting from the HEAD "
             "commits. 0 corresponds to HEAD only commits, 1 to HEAD and HEAD~1, 2 to HEAD, HEAD~1"
             " and HEAD~2, etc. With `--dzhigurda -1` we keep all possible commits for each "
             "document.")
