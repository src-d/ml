import os


VENDOR = "source{d}"
BACKEND = "gcs"
BACKEND_ARGS = "bucket=models.cdn.sourced.tech"
INDEX_REPO = "https://github.com/src-d/models"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "source{d}")
