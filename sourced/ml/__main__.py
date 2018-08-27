import argparse
import logging
import os
import sys

from modelforge.logs import setup_logging

from sourced.ml import extractors
from sourced.ml.transformers import Moder
from sourced.ml import cmd
from sourced.ml.cmd import args
from sourced.ml.cmd.run_swivel import mirror_tf_args
from sourced.ml.utils import install_bigartm, add_spark_args


def get_parser() -> argparse.ArgumentParser:
    """
    Creates the cmdline argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=args.ArgumentDefaultsHelpFormatterNoNone)
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")
    # Create and construct subparsers
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    def add_parser(name, help_message):
        return subparsers.add_parser(
            name, help=help_message, formatter_class=args.ArgumentDefaultsHelpFormatterNoNone)

    # ------------------------------------------------------------------------
    preprocessing_parser = subparsers.add_parser(
        "preprocrepos", help="Convert siva to parquet files with extracted information.")
    preprocessing_parser.set_defaults(handler=cmd.preprocess_repos)
    preprocessing_parser.add_argument("-x", "--mode", choices=Moder.Options.__all__,
                                      default="file", help="What to extract from repositories.")
    args.add_repo2_args(preprocessing_parser)
    preprocessing_parser.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the parquet files with bag batches.")
    default_fields = ("blob_id", "repository_id", "content", "path", "commit_hash", "uast", "lang")
    preprocessing_parser.add_argument(
        "-f", "--fields", nargs="+", default=default_fields,
        help="Fields to save.")
    # ------------------------------------------------------------------------
    repos2bow_parser = add_parser(
        "repos2bow", "Convert source code to the bag-of-words model.")
    repos2bow_parser.set_defaults(handler=cmd.repos2bow)
    args.add_df_args(repos2bow_parser)
    args.add_repo2_args(repos2bow_parser)
    args.add_feature_args(repos2bow_parser)
    args.add_bow_args(repos2bow_parser)
    args.add_repartitioner_arg(repos2bow_parser)
    args.add_cached_index_arg(repos2bow_parser)
    # ------------------------------------------------------------------------
    repos2bow_index_parser = add_parser(
        "repos2bow_index", "Creates the index, quant and docfreq model of the bag-of-words model.")
    repos2bow_index_parser.set_defaults(handler=cmd.repos2bow_index)
    args.add_df_args(repos2bow_index_parser)
    args.add_repo2_args(repos2bow_index_parser)
    args.add_feature_args(repos2bow_index_parser)
    args.add_repartitioner_arg(repos2bow_index_parser)
    args.add_cached_index_arg(repos2bow_index_parser, create=True)
    # ------------------------------------------------------------------------
    repos2df_parser = add_parser(
        "repos2df", "Calculate document frequencies of features extracted from source code.")
    repos2df_parser.set_defaults(handler=cmd.repos2df)
    args.add_df_args(repos2df_parser)
    args.add_repo2_args(repos2df_parser)
    args.add_feature_args(repos2df_parser)
    # ------------------------------------------------------------------------
    repos2ids_parser = subparsers.add_parser(
        "repos2ids", help="Convert source code to a bag of identifiers.")
    repos2ids_parser.set_defaults(handler=cmd.repos2ids)
    args.add_repo2_args(repos2ids_parser)
    args.add_split_stem_arg(repos2ids_parser)
    args.add_repartitioner_arg(repos2ids_parser)
    repos2ids_parser.add_argument(
        "-o", "--output", required=True,
        help="[OUT] output path to the CSV file with identifiers.")
    repos2ids_parser.add_argument(
        "--idfreq", action="store_true",
        help="Adds identifier frequencies to the output CSV file."
             "num_repos is the number of repositories where the identifier appears in."
             "num_files is the number of files where the identifier appears in."
             "num_occ is the total number of occurences of the identifier.")
    # ------------------------------------------------------------------------
    repos2coocc_parser = add_parser(
        "repos2coocc", "Convert source code to the sparse co-occurrence matrix of identifiers.")
    repos2coocc_parser.set_defaults(handler=cmd.repos2coocc)
    args.add_df_args(repos2coocc_parser)
    args.add_repo2_args(repos2coocc_parser)
    args.add_split_stem_arg(repos2coocc_parser)
    args.add_repartitioner_arg(repos2coocc_parser)
    repos2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the Cooccurrences model.")

    # ------------------------------------------------------------------------
    repos2roles_and_ids = add_parser(
        "repos2roleids", "Converts a UAST to a list of pairs, where pair is a role and "
        "identifier. Role is merged generic roles where identifier was found.")
    repos2roles_and_ids.set_defaults(handler=cmd.repos2roles_and_ids)
    args.add_repo2_args(repos2roles_and_ids)
    args.add_split_stem_arg(repos2roles_and_ids)
    repos2roles_and_ids.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the directory where spark should store the result. "
             "Inside the direcory you find result is csv format, status file and sumcheck files.")
    # ------------------------------------------------------------------------
    repos2identifier_distance = add_parser(
        "repos2id_distance", "Converts a UAST to a list of identifier pairs "
                             "and distance between them.")
    repos2identifier_distance.set_defaults(handler=cmd.repos2id_distance)
    args.add_repo2_args(repos2identifier_distance)
    args.add_split_stem_arg(repos2identifier_distance)
    repos2identifier_distance.add_argument(
        "-t", "--type", required=True, choices=extractors.IdentifierDistance.DistanceType.All,
        help="Distance type.")
    repos2identifier_distance.add_argument(
        "--max-distance", default=extractors.IdentifierDistance.DEFAULT_MAX_DISTANCE, type=int,
        help="Maximum distance to save.")
    repos2identifier_distance.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the directory where spark should store the result. "
             "Inside the direcory you find result is csv format, status file and sumcheck files.")
    # ------------------------------------------------------------------------
    repos2id_sequence = add_parser(
        "repos2idseq", "Converts a UAST to sequence of identifiers sorted by order of appearance.")
    repos2id_sequence.set_defaults(handler=cmd.repos2id_sequence)
    args.add_repo2_args(repos2id_sequence)
    args.add_split_stem_arg(repos2id_sequence)
    repos2id_sequence.add_argument(
        "--skip-docname", default=False, action="store_true",
        help="Do not save document name in CSV file, only identifier sequence.")
    repos2id_sequence.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the directory where spark should store the result. "
             "Inside the direcory you find result is csv format, status file and sumcheck files.")
    # ------------------------------------------------------------------------
    preproc_parser = add_parser(
        "id2vec-preproc", "Convert a sparse co-occurrence matrix to the Swivel shards.")
    preproc_parser.set_defaults(handler=cmd.id2vec_preprocess)
    args.add_df_args(preproc_parser)
    preproc_parser.add_argument("-s", "--shard-size", default=4096, type=int,
                                help="The shard (submatrix) size.")
    preproc_parser.add_argument(
        "-i", "--input",
        help="Concurrence model produced by repos2coocc.")
    preproc_parser.add_argument("-o", "--output", required=True, help="Output directory.")
    # ------------------------------------------------------------------------
    train_parser = add_parser(
        "id2vec-train", "Train identifier embeddings using Swivel.")
    train_parser.set_defaults(handler=cmd.run_swivel)
    mirror_tf_args(train_parser)
    # ------------------------------------------------------------------------
    id2vec_postproc_parser = add_parser(
        "id2vec-postproc",
        "Combine row and column embeddings produced by Swivel and write them to an .asdf.")
    id2vec_postproc_parser.set_defaults(handler=cmd.id2vec_postprocess)
    id2vec_postproc_parser.add_argument(
        "-i", "--swivel-data", required=True,
        help="Folder with swivel row and column embeddings data. "
             "You can get it using id2vec_train subcommand.")
    id2vec_postproc_parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for Id2Vec model.")
    # ------------------------------------------------------------------------
    id2vec_project_parser = add_parser(
        "id2vec-project", "Present id2vec model in Tensorflow Projector.")
    id2vec_project_parser.set_defaults(handler=cmd.id2vec_project)
    args.add_df_args(id2vec_project_parser, required=False)
    id2vec_project_parser.add_argument("-i", "--input", required=True,
                                       help="id2vec model to present.")
    id2vec_project_parser.add_argument("-o", "--output", required=True,
                                       help="Projector output directory.")
    id2vec_project_parser.add_argument("--no-browser", action="store_true",
                                       help="Do not open the browser.")
    # ------------------------------------------------------------------------
    train_id_split_parser = add_parser(
        "train-id-split", "Train a neural network to split identifiers.")
    train_id_split_parser.set_defaults(handler=cmd.train_id_split)
    # common arguments for CNN/RNN models
    train_id_split_parser.add_argument("-i", "--input", required=True,
                                       help="Path to the input data in CSV format:"
                                            "num_files,num_occ,num_repos,token,token_split")
    train_id_split_parser.add_argument("-e", "--epochs", type=int, default=10,
                                       help="Number of training epochs. The more the better"
                                            "but the training time is proportional.")
    train_id_split_parser.add_argument("-b", "--batch-size", type=int, default=500,
                                       help="Batch size. Higher values better utilize GPUs"
                                            "but may harm the convergence.")
    train_id_split_parser.add_argument("-l", "--length", type=int, default=40,
                                       help="RNN sequence length.")
    train_id_split_parser.add_argument("-o", "--output", required=True,
                                       help="Path to store the trained model.")
    train_id_split_parser.add_argument("-t", "--test-ratio", type=float, default=0.2,
                                       help="Fraction of the dataset to use for evaluation.")
    train_id_split_parser.add_argument("-p", "--padding", default="post", choices=("pre", "post"),
                                       help="Wether to pad before or after each sequence.")
    train_id_split_parser.add_argument("--optimizer", default="Adam", choices=("RMSprop", "Adam"),
                                       help="Algorithm to use as an optimizer for the neural net.")
    train_id_split_parser.add_argument("--lr", default=0.001, type=float,
                                       help="Initial learning rate.")
    train_id_split_parser.add_argument("--final-lr", default=0.00001, type=float,
                                       help="Final learning rate. The decrease from "
                                            "the initial learning rate is done linearly.")
    train_id_split_parser.add_argument("--samples-before-report", type=int, default=5*10**6,
                                       help="Number of samples between each validation report"
                                            "and training updates.")
    train_id_split_parser.add_argument("--val-batch-size", type=int, default=2000,
                                       help="Batch size for validation."
                                            "It can be increased to speed up the pipeline but"
                                            "it proportionally increases the memory consumption.")
    train_id_split_parser.add_argument("--seed", type=int, default=1989,
                                       help="Random seed.")
    train_id_split_parser.add_argument("--devices", default="0",
                                       help="Device(s) to use. '-1' means CPU.")
    train_id_split_parser.add_argument("--csv-identifier", default=3,
                                       help="Column name in the CSV file for the raw identifier.")
    train_id_split_parser.add_argument("--csv-identifier-split", default=4,
                                       help="Column name in the CSV file for the splitted"
                                            "identifier.")
    train_id_split_parser.add_argument("--include-csv-header", action="store_true",
                                       help="Treat the first line of the input CSV as a regular"
                                            "line.")
    train_id_split_parser.add_argument("--model", type=str, choices=("RNN", "CNN"), required=True,
                                       help="Neural Network model to use to learn the identifier"
                                            "splitting task.")
    train_id_split_parser.add_argument("-s", "--stack", default=2, type=int,
                                       help="Number of layers stacked on each other.")
    # RNN specific arguments
    train_id_split_parser.add_argument("--type-cell", default="LSTM",
                                       choices=("GRU", "LSTM", "CuDNNLSTM", "CuDNNGRU"),
                                       help="Recurrent layer type to use.")
    train_id_split_parser.add_argument("-n", "--neurons", default=256, type=int,
                                       help="Number of neurons on each layer.")
    # CNN specific arguments
    train_id_split_parser.add_argument("-f", "--filters", default="64,32,16,8",
                                       help="Number of filters for each kernel size.")
    train_id_split_parser.add_argument("-k", "--kernel-sizes", default="2,4,8,16",
                                       help="Sizes for sliding windows.")
    train_id_split_parser.add_argument("--dim-reduction", default=32, type=int,
                                       help="Number of 1-d kernels to reduce dimensionality"
                                            "after each layer.")
    # ------------------------------------------------------------------------
    bow2vw_parser = add_parser(
        "bow2vw", "Convert a bag-of-words model to the dataset in Vowpal Wabbit format.")
    bow2vw_parser.set_defaults(handler=cmd.bow2vw)
    bow2vw_parser.add_argument(
        "--bow", help="URL or path to a bag-of-words model.")
    bow2vw_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings.")
    bow2vw_parser.add_argument(
        "-o", "--output", required=True, help="Path to the output file.")
    # ------------------------------------------------------------------------
    bigartm_postproc_parser = add_parser(
        "bigartm2asdf", "Convert a human-readable BigARTM model to Modelforge format.")
    bigartm_postproc_parser.set_defaults(handler=cmd.bigartm2asdf)
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
    merge_df = add_parser("merge-df", "Merge DocumentFrequencies models to a single one.")
    merge_df.set_defaults(handler=cmd.merge_df)
    args.add_min_docfreq(merge_df)
    args.add_vocabulary_size_arg(merge_df)
    merge_df.add_argument(
        "-o", "--output", required=True,
        help="Path to the merged DocumentFrequencies model.")
    merge_df.add_argument(
        "-i", "--input", required=True, nargs="+",
        help="DocumentFrequencies models input files."
             "Use `-i -` to read input files from stdin.")
    merge_df.add_argument(
        "--ordered", action="store_true", default=False,
        help="Save OrderedDocumentFrequencies. "
             "If not specified DocumentFrequencies model will be saved")
    # ------------------------------------------------------------------------
    merge_coocc = add_parser("merge-coocc", "Merge several Cooccurrences models together.")
    merge_coocc.set_defaults(handler=cmd.merge_coocc)
    add_spark_args(merge_coocc)
    merge_coocc.add_argument(
        "-o", "--output", required=True,
        help="Path to the merged Cooccurrences model.")
    merge_coocc.add_argument(
        "-i", "--input", required=True,
        help="Cooccurrences models input files."
             "Use `-i -` to read input files from stdin.")
    merge_coocc.add_argument(
        "--docfreq", required=True,
        help="[IN] Specify OrderedDocumentFrequencies model. "
             "Identifiers that are not present in the model will be ignored.")
    merge_coocc.add_argument(
        "--no-spark", action="store_true", default=False,
        help="Use the local reduction instead of PySpark. "
             "Can be faster and consume less memory if the data fits into RAM.")
    # ------------------------------------------------------------------------
    merge_bow = add_parser("merge-bow", "Merge BOW models to a single one.")
    merge_bow.set_defaults(handler=cmd.merge_bow)
    merge_bow.add_argument(
        "-i", "--input", required=True, nargs="+",
        help="BOW models input files."
             "Use `-i -` to read input files from stdin.")
    merge_bow.add_argument(
        "-o", "--output", required=True,
        help="Path to the merged BOW model.")
    merge_bow.add_argument(
        "-f", "--features", nargs="+",
        choices=[ex.NAME for ex in extractors.__extractors__.values()],
        default=None, help="To keep only specific features, if not specified all will be kept.")
    # ------------------------------------------------------------------------
    id2role_eval = add_parser("id2role-eval",
                              "Compare the embeddings quality on role prediction problem.")
    id2role_eval.set_defaults(handler=cmd.id2role_eval)
    id2role_eval.add_argument(
        "-m", "--models", required=True, nargs="+",
        help="Id2Vec models to compare."
             "Use `-i -` to read input files from stdin.")
    id2role_eval.add_argument(
        "-d", "--dataset", required=True,
        help="Dataset directory. You can collect dataset via repos2roleids command.")
    id2role_eval.add_argument(
        "-s", "--seed", default=420,
        help="Random seed for reproducible results.")
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
