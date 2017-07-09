import logging
import sys

from clint.textui import progress


def progress_bar(enumerable, logger, **kwargs):
    """
    Shows the progress bar in the terminal, if the logging level matches and we are interactive.
    :param enumerable: The iterator of which we indicate the progress.
    :param logger: The bound logging.Logger.
    :param kwargs: Keyword arguments to pass to clint.textui.progress.bar.
    :return: The wrapped iterator.
    """
    if not logger.isEnabledFor(logging.INFO) or sys.stdin.closed or not sys.stdin.isatty():
        return enumerable
    return progress.bar(enumerable, **kwargs)
