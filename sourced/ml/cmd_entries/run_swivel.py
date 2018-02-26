import argparse
import logging
import warnings

try:
    from sourced.ml.algorithms import swivel as swivel

    def mirror_tf_args(parser: argparse.ArgumentParser):
        """
        Copies the command line flags registered in swivel.py to an :class:
        `argparse.ArgumentParser`.
        :param parser: object to copy flags to.
        :return: None
        """
        types = {"string": str, "int": int, "float": float, "bool": bool}
        for flag in swivel.FLAGS.__dict__["__wrapped"].__dict__["__flags"].values():
            parser.add_argument(
                "--" + flag.name, default=flag.default, type=types[flag.flag_type()],
                help=flag.help)
except ImportError as e:
    warnings.warn("Tensorflow is not installed, dependent functionality is unavailable.")

    def mirror_tf_args(parser: argparse.ArgumentParser):
        """Dummy handler to avoid errors in main."""
        pass


def run_swivel(args: argparse.Namespace):
    """
    Trains the Swivel model. Wraps swivel.py, taken from
    https://github.com/src-d/tensorflow-swivel/blob/master/swivel.py

    :param args: :class:`argparse.Namespace` identical to \
                 :class:`tf.app.flags`.
    :return: None
    """
    swivel.FLAGS = args
    logging.getLogger("tensorflow").handlers.clear()
    swivel.main(args)
