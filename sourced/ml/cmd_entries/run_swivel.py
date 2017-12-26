import logging

from sourced.ml.algorithms import swivel as swivel


def run_swivel(args):
    """
    Trains the Swivel model. Wraps swivel.py, adapted from
    https://github.com/vmarkovtsev/models/blob/master/swivel/swivel.py

    :param args: :class:`argparse.Namespace` identical to \
                 :class:`tf.app.flags`.
    :return: None
    """
    swivel.FLAGS = args
    logging.getLogger("tensorflow").handlers.clear()
    swivel.main(args)
