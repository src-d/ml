from datetime import datetime

import ast2vec


def generate_meta(name, *deps):
    return {
        "model": name,
        "dependencies": [d.meta for d in deps],
        "version": ast2vec.__version__,
        "created_at": datetime.now()
    }
