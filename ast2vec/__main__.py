import argparse
import logging
import os
import sys

from ast2vec.enry import install_enry
from ast2vec.id_embedding import preprocess, run_swivel, postprocess, swivel
from ast2vec.repo2coocc import repo2coocc_entry, repos2coocc_entry
from ast2vec.repo2nbow import repo2nbow_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")
    subparsers = parser.add_subparsers(help="Commands", dest="command")
    repo2nbow_parser = subparsers.add_parser(
        "repo2nbow", help="Produce the nBOW from a Git repository.")
    repo2nbow_parser.set_defaults(handler=repo2nbow_entry)
    repo2nbow_parser.add_argument(
        "-r", "--repository", required=True,
        help="URL or path to a Git repository.")
    repo2nbow_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings.")
    repo2nbow_parser.add_argument(
        "--df", help="URL or path to the document frequencies.")
    repo2nbow_parser.add_argument(
        "--linguist", help="Path to github/linguist-like executable.")
    repo2nbow_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.")

    repo2coocc_parser = subparsers.add_parser(
        "repo2coocc", help="Produce the co-occurrence matrix from a Git "
                           "repository.")
    repo2coocc_parser.set_defaults(handler=repo2coocc_entry)
    repo2coocc_parser.add_argument(
        "-r", "--repository", required=True,
        help="URL or path to a Git repository.")
    repo2coocc_parser.add_argument(
        "--linguist", help="Path to github/linguist-like executable.")
    repo2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="Output path where .npz result will be stored.")
    repo2coocc_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.")

    repos2coocc_parser = subparsers.add_parser(
        "repos2coocc", help="Produce the co-occurrence matrix from a list of "
                            "Git repositories.")
    repos2coocc_parser.set_defaults(handler=repos2coocc_entry)
    repos2coocc_parser.add_argument(
        "-i", "--input", required=True,
        help="list of repositories or path to file with list of repositories.")
    repos2coocc_parser.add_argument(
        "--linguist", help="Path to github/linguist-like executable.")
    repos2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="Output folder where .npz result will be stored.")
    repos2coocc_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.")

    preproc_parser = subparsers.add_parser(
        "preproc", help="Convert co-occurrence CSR matrices to Swivel "
                        "dataset.")
    preproc_parser.set_defaults(handler=preprocess)
    preproc_parser.add_argument(
        "-o", "--output", required=True, help="The output directory.")
    preproc_parser.add_argument(
        "-v", "--vocabulary-size", default=1 << 17,
        help="The final vocabulary size. Only the most frequent words will be"
             "left.")
    preproc_parser.add_argument("-s", "--shard-size", default=4096,
                                help="The shard (submatrix) size.")
    preproc_parser.add_argument(
        "--df", default=None,
        help="Path to the calculated document frequencies in npz format "
             "(DF in TF-IDF).")
    preproc_parser.add_argument(
        "input", nargs="+",
        help="Pickled scipy.sparse matrices. If it is a directory, all files "
             "inside are read.")

    train_parser = subparsers.add_parser(
        "train", help="Train identifier embeddings.")
    train_parser.set_defaults(handler=run_swivel)
    del train_parser._action_groups[train_parser._action_groups.index(
        train_parser._optionals)]
    train_parser._optionals = swivel.flags._global_parser._optionals
    train_parser._action_groups.append(train_parser._optionals)
    train_parser._actions = swivel.flags._global_parser._actions

    postproc_parser = subparsers.add_parser(
        "postproc", help="Combine row and column embeddings together and "
                         "write them to an .npz.")
    postproc_parser.set_defaults(handler=postprocess)
    postproc_parser.add_argument("swivel_output_directory")
    postproc_parser.add_argument("npz")

    enry_parser = subparsers.add_parser(
        "enry", help="Install src-d/enry to the current working directory.")
    enry_parser.set_defaults(handler=install_enry)
    enry_parser.add_argument(
        "--tempdir",
        help="Store intermediate files in this directory instead of /tmp.")
    enry_parser.add_argument("--output", default=os.getcwd(),
                             help="Output directory.")

    args = parser.parse_args()
    logging.basicConfig(level=logging._nameToLevel[args.log_level])
    return args.handler(args)

if __name__ == "__main__":
    sys.exit(main())
