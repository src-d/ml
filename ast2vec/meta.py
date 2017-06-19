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
