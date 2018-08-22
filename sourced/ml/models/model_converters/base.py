import logging
import multiprocessing
import os
from typing import Union, List

from modelforge import Model
from modelforge.progress_bar import progress_bar

from sourced.ml.utils.pickleable_logger import PickleableLogger


class Model2Base(PickleableLogger):
    """
    Base class for model -> model conversions.
    """
    MODEL_FROM_CLASS = None
    MODEL_TO_CLASS = None

    def __init__(self, num_processes: int=0,
                 log_level: int=logging.DEBUG, overwrite_existing: bool=True):
        """
        Initializes a new instance of Model2Base class.

        :param num_processes: The number of processes to execute for conversion.
        :param log_level: Logging verbosity level.
        :param overwrite_existing: Rewrite existing models or skip them.
        """
        super().__init__(log_level=log_level)
        self.num_processes = multiprocessing.cpu_count() if num_processes == 0 else num_processes
        self.overwrite_existing = overwrite_existing

    def convert(self, models_path: List[str], destdir: str) -> int:
        """
        Performs the model -> model conversion. Runs the conversions in a pool of processes.

        :param models_path: List of Models path.
        :param destdir: The directory where to store the models. The directory structure is \
                        preserved.
        :return: The number of converted files.
        """
        files = list(models_path)
        self._log.info("Found %d files", len(files))
        if not files:
            return 0
        queue_in = multiprocessing.Manager().Queue()
        queue_out = multiprocessing.Manager().Queue(1)
        processes = [multiprocessing.Process(target=self._process_entry,
                                             args=(i, destdir, queue_in, queue_out))
                     for i in range(self.num_processes)]
        for p in processes:
            p.start()
        for f in files:
            queue_in.put(f)
        for _ in processes:
            queue_in.put(None)
        failures = 0
        for _ in progress_bar(files, self._log, expected_size=len(files)):
            filename, ok = queue_out.get()
            if not ok:
                failures += 1
        for p in processes:
            p.join()
        self._log.info("Finished, %d failed files", failures)
        return len(files) - failures

    def convert_model(self, model: Model) -> Union[Model, None]:
        """
        This must be implemented in the child classes.

        :param model: The model instance to convert.
        :return: The converted model instance or None if it is not needed.
        """
        raise NotImplementedError

    def finalize(self, index: int, destdir: str):
        """
        Called for each worker in the end of the processing.

        :param index: Worker's index.
        :param destdir: The directory where to store the models.
        """
        pass

    def _process_entry(self, index, destdir, queue_in, queue_out):
        while True:
            filepath = queue_in.get()
            if filepath is None:
                break
            try:
                model_path = os.path.join(destdir, os.path.split(filepath)[1])
                if os.path.exists(model_path):
                    if self.overwrite_existing:
                        self._log.warning(
                            "Model %s already exists, but will be overwrite. If you want to "
                            "skip existing models use --disable-overwrite flag", model_path)
                    else:
                        self._log.warning("Model %s already exists, skipping.", model_path)
                        queue_out.put((filepath, True))
                        continue
                model_from = self.MODEL_FROM_CLASS(log_level=self._log.level).load(filepath)
                model_to = self.convert_model(model_from)
                if model_to is not None:
                    dirs = os.path.dirname(model_path)
                    if dirs:
                        os.makedirs(dirs, exist_ok=True)
                    model_to.save(model_path, deps=model_to.meta["dependencies"])
            except:  # noqa
                self._log.exception("%s failed", filepath)
                queue_out.put((filepath, False))
            else:
                queue_out.put((filepath, True))
        self.finalize(index, destdir)

    def _get_log_name(self):
        return "%s2%s" % (self.MODEL_FROM_CLASS.NAME, self.MODEL_TO_CLASS.NAME)
