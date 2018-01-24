import argparse
import logging
import os
import sys

from modelforge.logs import setup_logging
from sourced.ml import extractors
from sourced.ml.algorithms import swivel  # to access FLAGS
from sourced.ml.cmd_entries import bigartm2asdf_entry, dump_model, projector_entry, bow2vw_entry, \
    run_swivel, postprocess_id2vec, preprocess_id2vec, repos2coocc_entry, repos2df_entry, \
    repos2bow_entry
from sourced.ml.utils import install_bigartm, add_engine_args


class ArgumentDefaultsHelpFormatterNoNone(argparse.ArgumentDefaultsHelpFormatter):
    """
    Pretty formatter of help message for arguments.
    It adds default value to the end if it is not None.
    """
    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def get_parser() -> argparse.ArgumentParser:
    """
    Creates the cmdline argument parser.
    """

    def add_default_args(my_parser):
        my_parser.add_argument(
            "-r", "--repositories", required=True,
            help="The path to the repositories.")
        my_parser.add_argument(
            "--min-docfreq", default=1, type=int,
            help="The minimum document frequency of each element.")
        my_parser.add_argument(
            "-l", "--languages", required=True, nargs="+", choices=("Java", "Python"),
            help="The programming languages to analyse.")

    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")

    # Create and construct subparsers
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    repos2bow_parser = subparsers.add_parser(
        "repos2bow", help="Convert source code to the Bag-Of-Words model.")
    repos2bow_parser.set_defaults(handler=repos2bow_entry)
    add_default_args(repos2bow_parser)
    add_engine_args(repos2bow_parser)
    repos2bow_parser.add_argument(
        "--docfreq", required=True,
        help="[OUT] The path to the (Ordered)DocumentFrequencies model.")
    repos2bow_parser.add_argument(
        "--bow", required=True,
        help="[OUT] The path to the Bag-Of-Words model.")
    repos2bow_parser.add_argument(
        "--vocabulary-size", default=10000000, type=int,
        help="The maximum vocabulary size.")
    repos2bow_parser.add_argument(
        "--ordered", action="store_true",
        help="Flag that specifies ordered or default document frequency model to create."
             "If you use default document frequency model you should use only one feature.")
    repos2bow_parser.add_argument(
        "-f", "--feature", nargs="+",
        choices=[ex.NAME for ex in extractors.__extractors__.values()],
        required=True, help="The feature extraction scheme to apply.")

    repos2df_parser = subparsers.add_parser(
        "repos2df", help="Convert source code to document frequency model.")
    repos2df_parser.set_defaults(handler=repos2df_entry)
    add_default_args(repos2df_parser)
    add_engine_args(repos2df_parser)
    repos2df_parser.add_argument(
        "--docfreq", required=True,
        help="[OUT] The path to the (Ordered)DocumentFrequencies model.")
    repos2df_parser.add_argument(
        "--vocabulary-size", default=10000000, type=int,
        help="The maximum vocabulary size.")
    repos2df_parser.add_argument(
        "--ordered", action="store_true",
        help="Flag that specifies ordered or default document frequency model to create."
             "If you use default document frequency model you should use only one feature.")
    repos2df_parser.add_argument(
        "-f", "--feature", nargs="+",
        choices=[ex.NAME for ex in extractors.__extractors__.values()],
        required=True, help="The feature extraction scheme to apply.")

    repo2coocc_parser = subparsers.add_parser(
        "repos2coocc", help="Produce the co-occurrence matrix from a Git repository.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    add_engine_args(repo2coocc_parser)

    add_default_args(repo2coocc_parser)
    repo2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output file.")

    repo2coocc_parser.set_defaults(handler=repos2coocc_entry)

    preproc_parser = subparsers.add_parser(
        "id2vec_preproc", help="Convert co-occurrence CSR matrix to Swivel dataset.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    preproc_parser.set_defaults(handler=preprocess_id2vec)
    preproc_parser.add_argument(
        "-v", "--vocabulary-size", default=1 << 17, type=int,
        help="The final vocabulary size. Only the most frequent words will be"
             "left.")
    preproc_parser.add_argument("-s", "--shard-size", default=4096, type=int,
                                help="The shard (submatrix) size.")
    preproc_parser.add_argument(
        "--docfreq", default=None,
        help="[IN] Path to the pre-calculated document frequencies in asdf format "
             "(DF in TF-IDF).")
    preproc_parser.add_argument(
        "-i", "--input",
        help="Concurrence model produced by repos2coocc.")
    preproc_parser.add_argument("-o", "--output", required=True, help="Output directory.")

    train_parser = subparsers.add_parser(
        "id2vec_train", help="Train identifier embeddings.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    train_parser.set_defaults(handler=run_swivel)
    del train_parser._action_groups[train_parser._action_groups.index(
        train_parser._optionals)]
    train_parser._optionals = swivel.flags._global_parser._optionals
    train_parser._action_groups.append(train_parser._optionals)
    train_parser._actions = swivel.flags._global_parser._actions
    train_parser._option_string_actions = \
        swivel.flags._global_parser._option_string_actions

    id2vec_postproc_parser = subparsers.add_parser(
        "id2vec_postproc",
        help="Combine row and column embeddings together and write them to an .asdf.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    id2vec_postproc_parser.set_defaults(handler=postprocess_id2vec)
    id2vec_postproc_parser.add_argument("--swivel-output-directory")
    id2vec_postproc_parser.add_argument("--result")

    id2vec_projector_parser = subparsers.add_parser(
        "id2vec_projector", help="Present id2vec model in Tensorflow Projector.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    id2vec_projector_parser.set_defaults(handler=projector_entry)
    id2vec_projector_parser.add_argument("-i", "--input", required=True,
                                         help="id2vec model to present.")
    id2vec_projector_parser.add_argument("-o", "--output", required=True,
                                         help="Projector output directory.")
    id2vec_projector_parser.add_argument("--df", help="docfreq model to pick the most significant "
                                                      "identifiers.")
    id2vec_projector_parser.add_argument("--no-browser", action="store_true",
                                         help="Do not open the browser.")

    bow2vw_parser = subparsers.add_parser(
        "bow2vw", help="Convert a bag-of-words model to the dataset in Vowpal Wabbit format.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    bow2vw_parser.set_defaults(handler=bow2vw_entry)
    bow2vw_parser.add_argument(
        "--bow", help="URL or path to a bag-of-words model.")
    bow2vw_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings.")
    bow2vw_parser.add_argument(
        "-o", "--output", required=True, help="Path to the output file.")

    bigartm_postproc_parser = subparsers.add_parser(
        "bigartm2asdf", help="Convert a readable BigARTM model to Modelforge format.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    bigartm_postproc_parser.set_defaults(handler=bigartm2asdf_entry)
    bigartm_postproc_parser.add_argument("input")
    bigartm_postproc_parser.add_argument("output")

    bigartm_parser = subparsers.add_parser(
        "bigartm", help="Install bigartm/bigartm to the current working directory.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    bigartm_parser.set_defaults(handler=install_bigartm)
    bigartm_parser.add_argument(
        "--tmpdir", help="Store intermediate files in this directory instead of /tmp.")
    bigartm_parser.add_argument("--output", default=os.getcwd(), help="Output directory.")

    dump_parser = subparsers.add_parser(
        "dump", help="Dump a model to stdout.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    dump_parser.set_defaults(handler=dump_model)
    dump_parser.add_argument(
        "input", help="Path to the model file, URL or UUID.")
    dump_parser.add_argument("--gcs", default=None, dest="gcs_bucket",
                             help="GCS bucket to use.")

    return parser


def main():
    """
    Creates all the argparse-rs and invokes the function from set_defaults().

    :return: The result of the function from set_defaults().
    """

    parser = get_parser()
    args = parser.parse_args()
    args.log_level = logging._nameToLevel[args.log_level]
    setup_logging(args.log_level)
    try:
        handler = args.handler
    except AttributeError:
        def print_usage(_):
            parser.print_usage()

        handler = print_usage
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
