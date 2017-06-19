import io
import json
import logging
import math
import os
import uuid

import asdf
from clint.textui import progress
import numpy
import requests
import scipy.sparse


class Model:
    NAME = None
    DEFAULT_FILE_NAME = "default"
    DEFAULT_FILE_EXT = ".asdf"
    DEFAULT_SOURCE = "https://google.cloud.link"
    DEFAULT_CACHE_DIR = None

    def __init__(self, source=None, cache_dir=None, log_level=logging.INFO):
        self._log = logging.getLogger(self.NAME)
        self._log.setLevel(log_level)
        if cache_dir is None:
            cache_dir = self.DEFAULT_CACHE_DIR
        try:
            uuid.UUID(source)
            is_uuid = True
        except (TypeError, ValueError):
            is_uuid = False
        model_id = self.DEFAULT_FILE_NAME if not is_uuid else source
        file_name = model_id + self.DEFAULT_FILE_EXT
        file_name = os.path.join(os.path.expanduser(cache_dir), file_name)
        if os.path.exists(file_name):
            source = file_name
        elif source is None or is_uuid:
            buffer = io.BytesIO()
            self._fetch(self.DEFAULT_SOURCE, buffer)
            config = json.loads(buffer.getvalue().decode("utf-8"))
            source = config["id2vec"][model_id]["url"]
        if source.startswith("http://") or source.startswith("https://"):
            self._fetch(source, file_name)
            source = file_name
        self._log.info("Reading %s...", source)
        model = asdf.open(source)
        tree = model.tree
        self._meta = tree["meta"]
        if self.NAME != self._meta["model"]:
            raise ValueError("The supplied model is of the wrong type: needed "
                             "%s, got %s." % (self.NAME, self._meta["model"]))
        self._load(tree)

    @property
    def meta(self):
        return self._meta

    def __str__(self):
        return str(self._meta)

    def _load(self, tree):
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
                
                
def merge_strings(list_of_strings):
    strings = numpy.array(["".join(list_of_strings).encode("utf-8")])
    offset = 0
    offsets = []
    for s in list_of_strings:
        offsets.append(offset)
        offset += len(s)
    offsets = numpy.array(offsets, dtype=numpy.uint32)
    return {"strings": strings, "offsets": offsets}


def split_strings(subtree):
    result = []
    strings = subtree["strings"][0].decode("utf-8")
    offsets = subtree["offsets"]
    for i, offset in enumerate(offsets):
        if i < offsets.shape[0] - 1:
            result.append(strings[offset:offsets[i + 1]])
        else:
            result.append(strings[offset:])
    return result


def disassemble_sparse_matrix(matrix):
    result = {
        "shape": matrix.shape,
        "format": matrix.getformat()
    }
    if isinstance(matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        result["data"] = matrix.data, matrix.indices, matrix.indptr
    elif isinstance(matrix, scipy.sparse.coo_matrix):
        result["data"] = matrix.data, (matrix.row, matrix.col)
    return result


def assemble_sparse_matrix(subtree):
    matrix_class = getattr(scipy.sparse, "%s_matrix" % subtree["format"])
    matrix = matrix_class(tuple(subtree["data"]), shape=subtree["shape"])
    return matrix
