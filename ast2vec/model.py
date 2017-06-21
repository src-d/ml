import io
import json
import logging
import math
import os
import shutil
import tempfile
import uuid

import asdf
from clint.textui import progress
import numpy
import requests
import scipy.sparse


class Model:
    """
    Base class for all the models.
    """

    NAME = None  #: Name of the model. Used as the logging domain, too.
    DEFAULT_NAME = "default"  #: When no uuid is specified, this is used.
    DEFAULT_FILE_EXT = ".asdf"  #: File extension of the model.
    DEFAULT_GCS_BUCKET = "datasets.sourced.tech"  #: GCS bucket where the models are stored.
    INDEX_FILE = "index.json"  #: Models repository index file name.
    CACHE_DIR_ROOT = os.path.join("~", ".source{d}")  #: Cache root path.

    def __init__(self, source=None, cache_dir=None, gcs_bucket=None,
                 log_level=logging.INFO):
        """
        Initializes a new Model instance.
        :param source: UUID, file system path or an URL; None means auto.
        :param cache_dir: The directory where to store the downloaded model.
        :param gcs_bucket: The name of the Google Cloud Storage bucket to use.
        :param log_level: The logging level applied to this instance.
        """
        self._log = logging.getLogger(self.NAME)
        self._log.setLevel(log_level)
        if cache_dir is None:
            if self.NAME is not None:
                cache_dir = os.path.join(self.CACHE_DIR_ROOT, self.NAME)
            else:
                cache_dir = tempfile.mkdtemp(prefix="ast2vec-")
        try:
            try:
                uuid.UUID(source)
                is_uuid = True
            except (TypeError, ValueError):
                is_uuid = False
            model_id = self.DEFAULT_NAME if not is_uuid else source
            file_name = model_id + self.DEFAULT_FILE_EXT
            file_name = os.path.join(os.path.expanduser(cache_dir), file_name)
            if os.path.exists(file_name):
                source = file_name
            elif source is None or is_uuid:
                buffer = io.BytesIO()
                self._fetch(self.compose_index_url(gcs_bucket), buffer)
                config = json.loads(buffer.getvalue().decode("utf8"))["models"]
                if self.NAME is not None:
                    source = config[self.NAME][model_id]
                    if not is_uuid:
                        source = config[self.NAME][source]
                else:
                    if not is_uuid:
                        raise ValueError("File path, URL or UUID is needed.")
                    for models in config.values():
                        if source in models:
                            source = models[source]
                            break
                    else:
                        raise FileNotFoundError("Model %s not found." % source)
                source = source["url"]
            if source.startswith("http://") or source.startswith("https://"):
                self._fetch(source, file_name)
                source = file_name
            self._log.info("Reading %s...", source)
            model = asdf.open(source)
            tree = model.tree
            self._meta = tree["meta"]
            if self.NAME != self._meta["model"] and self.NAME is not None:
                raise ValueError(
                    "The supplied model is of the wrong type: needed "
                    "%s, got %s." % (self.NAME, self._meta["model"]))

            self._load(tree)
        finally:
            if self.NAME is None:
                shutil.rmtree(cache_dir)

    @property
    def meta(self):
        """
        Metadata dictionary: when was created, uuid, engine version, etc.
        """
        return self._meta

    def __str__(self):
        return str(self._meta)

    @classmethod
    def compose_index_url(cls, gcs=None):
        return "https://storage.googleapis.com/%s/%s" % (
            gcs if gcs else cls.DEFAULT_GCS_BUCKET, cls.INDEX_FILE)

    def _load(self, tree):
        raise NotImplementedError()

    def _fetch(self, url, where, chunk_size=65536):
        self._log.info("Fetching %s...", url)
        r = requests.get(url, stream=True)
        f = open(where, "wb") if isinstance(where, str) else where
        try:
            total_length = int(r.headers.get("content-length"))
            num_chunks = math.ceil(total_length / chunk_size)
            if num_chunks == 1:
                f.write(r.content)
            else:
                for chunk in progress.bar(
                        r.iter_content(chunk_size=chunk_size),
                        expected_size=num_chunks):
                    if chunk:
                        f.write(chunk)
        finally:
            if isinstance(where, str):
                f.close()
                
                
def merge_strings(list_of_strings):
    """
    Packs the list of strings into two arrays: the concatenated chars and the
    individual string offsets. :func:`split_strings()` does the inverse.
    :param list_of_strings: The list of strings to pack.
    :return: dict with "strings" and "offsets" arrays.
    """
    strings = numpy.array(["".join(list_of_strings).encode("utf-8")])
    offset = 0
    offsets = []
    for s in list_of_strings:
        offsets.append(offset)
        offset += len(s)
    offsets = numpy.array(offsets, dtype=numpy.uint32)
    return {"strings": strings, "offsets": offsets}


def split_strings(subtree):
    """
    Produces the list of strings from the dictionary with concatenated chars
    and offsets. Opposite to :func:`merge_strings()`.
    :param subtree: The dict with "strings" and "offsets".
    :return: list of strings.
    """
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
    """
    Transforms a scipy.sparse matrix into the serializable collection of numpy
    arrays. :func:`assemble_sparse_matrix()` does the inverse.
    :param matrix: scipy.sparse matrix; csr, csc and coo formats are supported.
    :return: dict with "shape", "format" and "data" - tuple of numpy arrays.
    """
    fmt = matrix.getformat()
    if fmt not in ("csr", "csc", "coo"):
        raise ValueError("Unsupported scipy.sparse matrix format: %s." % fmt)
    result = {
        "shape": matrix.shape,
        "format": fmt
    }
    if isinstance(matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        result["data"] = matrix.data, matrix.indices, matrix.indptr
    elif isinstance(matrix, scipy.sparse.coo_matrix):
        result["data"] = matrix.data, (matrix.row, matrix.col)
    return result


def assemble_sparse_matrix(subtree):
    """
    Transforms a dictionary with "shape", "format" and "data" into the
    scipy.sparse matrix. Opposite to :func:`disassemble_sparse_matrix()`.
    :param subtree: dict which describes the scipy.sparse matrix.
    :return: scipy.sparse matrix of the specified format.
    """
    matrix_class = getattr(scipy.sparse, "%s_matrix" % subtree["format"])
    matrix = matrix_class(tuple(subtree["data"]), shape=subtree["shape"])
    return matrix
