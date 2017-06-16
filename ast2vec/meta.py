from datetime import datetime

from ast2vec import __version__


def generate_meta(name, *deps):
    return {
        "model": name,
        "dependencies": [d.meta for d in deps],
        "version": __version__,
        "created_at": datetime.now()
    }
