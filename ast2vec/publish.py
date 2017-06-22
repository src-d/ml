import io
import json
import logging
import math
import os
import time
import uuid

import asdf
from clint.textui import progress
from dateutil.parser import parse as parse_datetime
from google.cloud.storage import Client
import requests

from ast2vec.meta import extract_index_meta
from ast2vec.model import Model


class FileReadTracker:
    """
    Wrapper around Python fileobj which records the file position and updates
    the console progressbar.
    """
    def __init__(self, file):
        self._file = file
        self._position = 0
        file.seek(0, 2)
        self._size = file.tell()
        self._progress = progress.Bar(expected_size=self._size)
        file.seek(0)

    @property
    def size(self):
        return self._size

    def read(self, size):
        result = self._file.read(size)
        self._position += len(result)
        self._progress.show(self._position)
        return result

    def tell(self):
        return self._position

    def done(self):
        self._progress.done()


def publish_model(args):
    """
    Pushes the model to Google Cloud Storage and updates the index file.

    :param args: :class:`argparse.Namespace` with "model", "gcs" and "force".
    :return: None if successful, 1 otherwise.
    """
    log = logging.getLogger("publish")
    log.info("Reading %s...", os.path.abspath(args.model))
    tree = asdf.open(args.model).tree
    meta = tree["meta"]
    log.info("Locking the bucket...")
    transaction = uuid.uuid4().hex.encode()
    client = Client()
    bucket = client.get_bucket(args.gcs)
    sentinel = bucket.blob("index.lock")
    locked = False
    while not locked:
        while sentinel.exists():
            log.warning("Failed to acquire the lock, waiting...")
            time.sleep(1)
        # At this step, several agents may think the lockfile does not exist
        try:
            sentinel.upload_from_string(transaction)
            # Only one agent succeeds to check this condition
            locked = sentinel.download_as_string() == transaction
        except:
            # GCS detects the changed-while-reading collision
            log.warning("Failed to acquire the lock, retrying...")
    try:
        blob = bucket.blob("models/%s/%s.asdf" % (meta["model"], meta["uuid"]))
        if blob.exists() and not args.force:
            log.error("Model %s already exists, aborted.", meta["uuid"])
            return 1
        log.info("Uploading %s from %s...", meta["model"],
                 os.path.abspath(args.model))
        with open(args.model, "rb") as fin:
            tracker = FileReadTracker(fin)
            try:
                blob.upload_from_file(
                    tracker, content_type="application/x-yaml",
                    size=tracker.size)
            finally:
                tracker.done()
        blob.make_public()
        model_url = blob.public_url
        log.info("Uploaded as %s", blob.path)
        log.info("Updating the models index...")
        blob = bucket.get_blob(Model.INDEX_FILE)
        index = json.loads(blob.download_as_string().decode("utf-8"))
        index["models"].setdefault(meta["model"], {})[meta["uuid"]] = \
            extract_index_meta(meta, model_url)
        if args.update_default:
            index["models"][meta["model"]][Model.DEFAULT_NAME] = meta["uuid"]
        blob.upload_from_string(json.dumps(index, indent=4, sort_keys=True))
        blob.make_public()
    finally:
        sentinel.delete()


def list_models(args):
    """
    Outputs the list of known models in the registry.

    :param args: :class:`argparse.Namespace` with "gcs".
    :return: None
    """
    r = requests.get(Model.compose_index_url(args.gcs), stream=True)
    content = r.content.decode("utf-8")
    try:
        index = json.loads(content)
    except json.decoder.JSONDecodeError:
        print(content)
        return 1
    for key, val in index["models"].items():
        print(key)
        default = None
        for mid, meta in val.items():
            if mid == "default":
                default = meta
                break
        for mid, meta in sorted(
                [m for m in val.items() if m[1] != default],
                key=lambda m: parse_datetime(m[1]["created_at"])):
            print("  %s %s" % ("*" if default == mid else " ", mid),
                  meta["created_at"])
