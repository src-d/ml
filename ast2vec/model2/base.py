from itertools import repeat
import logging
import multiprocessing
import os
from pathlib import Path
import threading

from modelforge import Model
from modelforge.progress_bar import progress_bar

from ast2vec.pickleable_logger import PickleableLogger


class Model2Base(PickleableLogger):
    """
    Base class for model -> model conversions.
    """
    MODEL_FROM_CLASS = None
    MODEL_TO_CLASS = None

    def __init__(self, num_processes: int=multiprocessing.cpu_count(),
                 log_level: int=logging.DEBUG):
        """
        Initializes a new instance of Model2Base class.

        :param num_processes: The number of processes to execute for conversion.
        :param log_level: Logging verbosity level.
        """
        super(Model2Base, self).__init__(log_level=log_level)
        self.num_processes = num_processes

    def convert(self, srcdir: str, destdir: str, pattern: str="**/*.asdf") -> int:
        """
        Performs the model -> model conversion. Runs the conversions in a pool of processes.

        :param srcdir: The directory to scan for the models.
        :param destdir: The directory where to store the models. The directory structure is \
                        preserved.
        :param pattern: glob pattern for the files.
        :return:
        """
        self._log.info("Scanning %s", srcdir)
        files = [str(p) for p in Path(srcdir).glob(pattern)]
        self._log.info("Found %d files", len(files))
        queue = multiprocessing.Manager().Queue(100500)

        def process_files():
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                pool.starmap(self._process_model,
                             zip(files, repeat(destdir), repeat(srcdir), repeat(queue)))

        mpthread = threading.Thread(target=process_files)
        mpthread.start()
        failures = 0
        for _ in progress_bar(files, self._log, expected_size=len(files)):
            filename, ok = queue.get()
            if not ok:
                failures += 1
        mpthread.join()
        self._log.info("Finished, %d failed files", failures)
        return len(files) - failures

    def convert_model(self, model: Model) -> Model:
        """
        This must be implemented in the child classes.

        :param model: The model instance to convert.
        :return: The converted model instance.
        """
        raise NotImplementedError

    def _get_log_name(self):
        return "%s2%s" % (self.MODEL_FROM_CLASS.NAME, self.MODEL_TO_CLASS.NAME)

    def _get_model_path(self, path):
        """
        By default, we name the converted files exactly the same.

        :param path: The path relative to ``srcdir``.
        :return: The target path for the converted model.
        """
        return path

    def _process_model(self, filename, srcdir, destdir, queue):
        try:
            model_from = self.MODEL_FROM_CLASS().load(filename)
            model_to = self.convert_model(model_from)
            model_path = self._get_model_path(os.path.relpath(filename, srcdir))
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_to.save(os.path.join(destdir, model_path))
        except:
            self._log.exception("%s failed", filename)
            queue.put((filename, False))
        else:
            queue.put((filename, True))
