import gzip
import io
import json
import logging
import math
import os

from clint.textui import progress
import numpy
import requests


class Dataset:
    LOG_NAME = None
    DEFAULT_FILE_NAME = "default.npz"
    DEFAULT_SOURCE = "https://google.cloud.link"
    DEFAULT_CACHE_DIR = None

    def __init__(self, source=None, cache_dir=None, log_level=logging.INFO):
        self._log = logging.getLogger(self.LOG_NAME)
        self._log.setLevel(log_level)
        if cache_dir is None:
            cache_dir = self.DEFAULT_CACHE_DIR
        default_file_name = os.path.join(os.path.expanduser(cache_dir),
                                         self.DEFAULT_FILE_NAME)
        if source is None:
            if os.path.exists(default_file_name):
                source = default_file_name
            else:
                buffer = io.BytesIO()
                self._fetch(self.DEFAULT_SOURCE, buffer)
                config = json.loads(buffer.getvalue().decode("utf-8"))
                source = config["id2vec"]["default"]["url"]
        if source.startswith("http://") or source.startswith("https://"):
            self._fetch(source, default_file_name)
            source = default_file_name
        self._log.info("Reading %s...", source)
        if source.endswith(".gz"):
            with gzip.open(source) as f:
                npz = numpy.load(f)
        else:
            npz = numpy.load(source)
        self._meta = npz["meta"]
        self._load(npz)

    @property
    def meta(self):
        return self._meta

    def __str__(self):
        return str(self._meta)

    def _load(self, npz):
        raise NotImplementedError()

    def _fetch(self, url, where, chunk_size=65536):
        self._log.info("Fetching %s...", url)
        r = requests.get(url, stream=True)
        f = open(where, "wb") if isinstance(where, str) else where
        try:
            total_length = int(r.headers.get("content-length"))
            num_chunks = math.ceil(total_length / chunk_size)
            for chunk in progress.bar(r.iter_content(chunk_size=chunk_size),
                                      expected_size=num_chunks):
                if chunk:
                    f.write(chunk)
        finally:
            if isinstance(where, str):
                f.close()
