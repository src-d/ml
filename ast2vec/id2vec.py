import io
import json
import logging
import math
import os

from clint.textui import progress
import numpy
import requests


class Id2Vec:
    DEFAULT_FILE_NAME = "default.npz"
    DEFAULT_SOURCE = "https://google.cloud.link"
    DEFAULT_CACHE_DIR = "~/.cache/source{d}/id2vec"

    def __init__(self, source=None, cache_dir=None, log_level=logging.INFO):
        self._log = logging.getLogger("id2vec")
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
        npz = numpy.load(source)
        self._embeddings = npz["embeddings"]
        self._tokens = npz["tokens"]
        self._log.info("Building the token index...")
        self._token2index = {w: i for i, w in enumerate(self._tokens)}

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def tokens(self):
        return self._tokens

    @property
    def token2index(self):
        return self._token2index

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
