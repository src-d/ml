from datetime import datetime
import uuid

import ast2vec


def generate_meta(name, *deps):
    return {
        "model": name,
        "uuid": str(uuid.uuid4()),
        "dependencies": [d.meta for d in deps],
        "version": ast2vec.__version__,
        "created_at": datetime.now()
    }


def _extract_index_meta_dependency(meta):
    return {
        "version": meta["version"],
        "uuid": meta["uuid"],
        "dependencies": [_extract_index_meta_dependency(m)
                         for m in meta["dependencies"]],
        "created_at": str(meta["created_at"]),
    }


def extract_index_meta(meta, model_url):
    return {
        "version": meta["version"],
        "url": model_url,
        "dependencies": [_extract_index_meta_dependency(m)
                         for m in meta["dependencies"]],
        "created_at": str(meta["created_at"]),
    }
