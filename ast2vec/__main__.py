import argparse
import sys

from ast2vec.id_embedding import preprocess, run_swivel, postprocess, swivel
from ast2vec.repo2nbow import repo2nbow2stdout


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Commands", dest="command")
    repo2nbow_parser = subparsers.add_parser(
        "repo2nbow", help="Produce the nBOW from a Git repository.")
    repo2nbow_parser.set_defaults(handler=repo2nbow2stdout)
    repo2nbow_parser.add_argument(
        "-r", "--repository", required=True,
        help="URL or path to a Git repository.")
    repo2nbow_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings.")
    repo2nbow_parser.add_argument(
        "--df", help="URL or path to the document frequencies.")
    repo2nbow_parser.add_argument(
        "--linguist", help="Path to github/linguist-like executable.")
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
    preproc_parser.add_argument("input", nargs="+",
                                help="Pickled scipy.sparse matrices.")
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
    postproc_parser.add_argument("swivel-output-directory")
    postproc_parser.add_argument("npz")
    args = parser.parse_args()
    args.handler(args)

if __name__ == "__main__":
    sys.exit(main())
