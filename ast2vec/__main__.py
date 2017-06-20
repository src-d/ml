import argparse
import codecs
import logging
import io
import os
import re
import sys

from ast2vec.enry import install_enry
from ast2vec.id_embedding import preprocess, run_swivel, postprocess, swivel
from ast2vec.publish import publish_model
from ast2vec.repo2coocc import repo2coocc_entry, repos2coocc_entry
from ast2vec.repo2nbow import repo2nbow_entry
from ast2vec.dump import dump_model


class ColorFormatter(logging.Formatter):
    """
    logging Formatter which prints messages with colors.
    """
    GREEN_MARKERS = [' ok', "ok:", 'finished', 'completed', 'ready',
                     'done', 'running', 'successful', 'saved']
    GREEN_RE = re.compile("|".join(GREEN_MARKERS))

    def formatMessage(self, record):
        level_color = "0"
        text_color = "0"
        fmt = ""
        if record.levelno <= logging.DEBUG:
            fmt = "\033[0;37m" + logging.BASIC_FORMAT + "\033[0m"
        elif record.levelno <= logging.INFO:
            level_color = "1;36"
            lmsg = record.message.lower()
            if self.GREEN_RE.search(lmsg):
                text_color = "1;32"
        elif record.levelno <= logging.WARNING:
            level_color = "1;33"
        elif record.levelno <= logging.CRITICAL:
            level_color = "1;31"
        if not fmt:
            fmt = "\033[" + level_color + \
                  "m%(levelname)s\033[0m:%(name)s:\033[" + text_color + \
                  "m%(message)s\033[0m"
        return fmt % record.__dict__


def setup_logging(level):
    """
    Makes stdout and stderr unicode friendly in case of misconfigured
    environments, initializes the logging and enables colored logs if it is
    appropriate.
    :param level: The logging level, can be either an int or a string.
    :return: None
    """
    if not isinstance(level, int):
        level = logging._nameToLevel[level]

    def ensure_utf8_stream(stream):
        if not isinstance(stream, io.StringIO):
            stream = codecs.getwriter("utf-8")(stream.buffer)
            stream.encoding = "utf-8"
        return stream

    sys.stdout, sys.stderr = (ensure_utf8_stream(s)
                              for s in (sys.stdout, sys.stderr))
    logging.basicConfig(level=level)
    if not sys.stdin.closed and sys.stdin.isatty():
        root = logging.getLogger()
        handler = root.handlers[0]
        handler.setFormatter(ColorFormatter())


def main():
    """
    Creates all the argparse-rs and invokes the function from set_defaults().
    :return: The result of the function from set_defaults().
    """
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
    repo2nbow_parser.add_argument(
        "-o", "--output", required=True,
        help="Output path where the .asdf will be stored.")

    repos2nbow_parser = subparsers.add_parser(
        "repos2nbow", help="Produce the nBOWs from a list of Git "
                           "repositories.")
    repos2nbow_parser.set_defaults(handler=repos2coocc_entry)
    repos2nbow_parser.add_argument(
        "-i", "--input", required=True, nargs="+",
        help="List of repositories or path to file with list of repositories.")
    repos2nbow_parser.add_argument(
        "--linguist", help="Path to github/linguist-like executable.")
    repos2nbow_parser.add_argument(
        "-o", "--output", required=True,
        help="Output folder where .asdf results will be stored.")
    repos2nbow_parser.add_argument(
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
        help="Output path where .asdf result will be stored.")
    repo2coocc_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.")

    repos2coocc_parser = subparsers.add_parser(
        "repos2coocc", help="Produce the co-occurrence matrix from a list of "
                            "Git repositories.")
    repos2coocc_parser.set_defaults(handler=repos2coocc_entry)
    repos2coocc_parser.add_argument(
        "-i", "--input", required=True, nargs="+",
        help="List of repositories or path to file with list of repositories.")
    repos2coocc_parser.add_argument(
        "--linguist", help="Path to github/linguist-like executable.")
    repos2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="Output folder where .asdf results will be stored.")
    repos2coocc_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.")

    preproc_parser = subparsers.add_parser(
        "preproc", help="Convert co-occurrence CSR matrices to Swivel "
                        "dataset.")
    preproc_parser.set_defaults(handler=preprocess)
    preproc_parser.add_argument(
        "-o", "--output", required=True, help="The output directory.")
    preproc_parser.add_argument(
        "-v", "--vocabulary-size", default=1 << 17, type=int,
        help="The final vocabulary size. Only the most frequent words will be"
             "left.")
    preproc_parser.add_argument("-s", "--shard-size", default=4096, type=int,
                                help="The shard (submatrix) size.")
    preproc_parser.add_argument(
        "--df", default=None,
        help="Path to the calculated document frequencies in asdf format "
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
    train_parser._option_string_actions = \
        swivel.flags._global_parser._option_string_actions

    postproc_parser = subparsers.add_parser(
        "postproc", help="Combine row and column embeddings together and "
                         "write them to an .asdf.")
    postproc_parser.set_defaults(handler=postprocess)
    postproc_parser.add_argument("swivel_output_directory")
    postproc_parser.add_argument("result")

    enry_parser = subparsers.add_parser(
        "enry", help="Install src-d/enry to the current working directory.")
    enry_parser.set_defaults(handler=install_enry)
    enry_parser.add_argument(
        "--tempdir",
        help="Store intermediate files in this directory instead of /tmp.")
    enry_parser.add_argument("--output", default=os.getcwd(),
                             help="Output directory.")

    dump_parser = subparsers.add_parser(
        "dump", help="Dump a model to stdout.")
    dump_parser.set_defaults(handler=dump_model)
    dump_parser.add_argument(
        "input", help="Path to the model file.")
    dump_parser.add_argument("-d", "--dependency", nargs="+",
                             help="Paths to the models which were used to "
                                  "generate the dumped model in the order "
                                  "they appear in the metadata.")

    publish_parser = subparsers.add_parser(
        "publish", help="Upload the model to the cloud and update the "
                        "registry.")
    publish_parser.set_defaults(handler=publish_model)
    publish_parser.add_argument(
        "model", help="The path to the model to publish.")
    publish_parser.add_argument("--gcs", default="datasets.sourced.tech",
                                help="GCS bucket to use.")
    publish_parser.add_argument("-u", "--update-default", action="store_true",
                                help="Update the models index.")
    publish_parser.add_argument("--force", action="store_true",
                                help="Overwrite existing models.")

    args = parser.parse_args()
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
