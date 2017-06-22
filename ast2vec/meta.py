from datetime import datetime
import uuid

import ast2vec


def generate_meta(name, *deps):
    """
    Creates the metadata tree for the given model name and the list of
    dependencies.

    :param name: The model's name.
    :param deps: The list of metas this model depends on.
    :return: dict with the metadata.
    """
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
    """
    Converts the metadata tree into a dict which is suitable for index.json.

    :param meta: tree["meta"] :class:`dict`.
    :param model_url: public URL of the model
    :return: converted dict.
    """
    return {
        "version": meta["version"],
        "url": model_url,
        "dependencies": [_extract_index_meta_dependency(m)
                         for m in meta["dependencies"]],
        "created_at": str(meta["created_at"]),
    }
