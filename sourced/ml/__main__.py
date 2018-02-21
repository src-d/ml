import argparse
import logging
import os
import sys

from modelforge.logs import setup_logging

from sourced.ml.cmd_entries import bigartm2asdf_entry, dump_model, projector_entry, bow2vw_entry, \
    run_swivel, postprocess_id2vec, preprocess_id2vec, repos2coocc_entry, repos2df_entry, \
    repos2bow_entry
from sourced.ml.cmd_entries.args import add_repo2_args, add_feature_args, add_vocabulary_size_arg
from sourced.ml.cmd_entries.repos2bow import add_bow_args
from sourced.ml.cmd_entries.run_swivel import mirror_tf_args
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

    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")
    # Create and construct subparsers
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    def add_parser(name, help):
        return subparsers.add_parser(
            name, help=help, formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # ------------------------------------------------------------------------
    repos2bow_parser = add_parser(
        "repos2bow", "Convert source code to the bag-of-words model.")
    repos2bow_parser.set_defaults(handler=repos2bow_entry)
    add_repo2_args(repos2bow_parser)
    add_engine_args(repos2bow_parser)
    add_bow_args(repos2bow_parser)
    add_feature_args(repos2bow_parser)
    # ------------------------------------------------------------------------
    repos2df_parser = add_parser(
        "repos2df", "Calculate document frequencies of features extracted from source code.")
    repos2df_parser.set_defaults(handler=repos2df_entry)
    add_repo2_args(repos2df_parser)
    add_engine_args(repos2df_parser)
    add_feature_args(repos2df_parser)
    # ------------------------------------------------------------------------
    repos2coocc_parser = add_parser(
        "repos2coocc", "Convert source code to the sparse co-occurrence matrix of identifiers.")
    add_engine_args(repos2coocc_parser)
    add_repo2_args(repos2coocc_parser, quant=False)
    repos2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the Cooccurrences model.")
    repos2coocc_parser.add_argument(
        "--split-stem", default=False, action="store_true",
        help="Split Tokens to parts (ThisIs_token -> ['this', 'is', 'token']).")
    repos2coocc_parser.set_defaults(handler=repos2coocc_entry)
    # ------------------------------------------------------------------------
    preproc_parser = add_parser(
        "id2vec_preproc", "Convert a sparse co-occurrence matrix to the Swivel shards.")
    preproc_parser.set_defaults(handler=preprocess_id2vec)
    add_vocabulary_size_arg(preproc_parser)
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
    # ------------------------------------------------------------------------
    train_parser = add_parser(
        "id2vec_train", "Train identifier embeddings using Swivel.")
    mirror_tf_args(train_parser)
    train_parser.set_defaults(handler=run_swivel)
    # ------------------------------------------------------------------------
    id2vec_postproc_parser = add_parser(
        "id2vec_postproc",
        "Combine row and column embeddings produced by Swivel and write them to an .asdf.")
    id2vec_postproc_parser.set_defaults(handler=postprocess_id2vec)
    id2vec_postproc_parser.add_argument(
        "-i", "--swivel-data", required=True,
        help="Folder with swivel batches input data. You can get it using repos2coocc subcommand.")
    id2vec_postproc_parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for embedding data.")
    # ------------------------------------------------------------------------
    id2vec_project_parser = add_parser(
        "id2vec_project", "Present id2vec model in Tensorflow Projector.")
    id2vec_project_parser.set_defaults(handler=projector_entry)
    id2vec_project_parser.add_argument("-i", "--input", required=True,
                                       help="id2vec model to present.")
    id2vec_project_parser.add_argument("-o", "--output", required=True,
                                       help="Projector output directory.")
    id2vec_project_parser.add_argument("--docfreq", help="docfreq model to pick the most "
                                                         "significant identifiers.")
    id2vec_project_parser.add_argument("--no-browser", action="store_true",
                                       help="Do not open the browser.")
    # ------------------------------------------------------------------------
    bow2vw_parser = add_parser(
        "bow2vw", "Convert a bag-of-words model to the dataset in Vowpal Wabbit format.")
    bow2vw_parser.set_defaults(handler=bow2vw_entry)
    bow2vw_parser.add_argument(
        "--bow", help="URL or path to a bag-of-words model.")
    bow2vw_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings.")
    bow2vw_parser.add_argument(
        "-o", "--output", required=True, help="Path to the output file.")
    # ------------------------------------------------------------------------
    bigartm_postproc_parser = add_parser(
        "bigartm2asdf", "Convert a human-readable BigARTM model to Modelforge format.")
    bigartm_postproc_parser.set_defaults(handler=bigartm2asdf_entry)
    bigartm_postproc_parser.add_argument("input")
    bigartm_postproc_parser.add_argument("output")
    # ------------------------------------------------------------------------
    bigartm_parser = add_parser(
        "bigartm", "Install bigartm/bigartm to the current working directory.")
    bigartm_parser.set_defaults(handler=install_bigartm)
    bigartm_parser.add_argument(
        "--tmpdir", help="Store intermediate files in this directory instead of /tmp.")
    bigartm_parser.add_argument("--output", default=os.getcwd(), help="Output directory.")
    # ------------------------------------------------------------------------
    dump_parser = add_parser("dump", "Dump a model to stdout.")
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
