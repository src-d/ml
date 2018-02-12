import gzip
import logging
from uuid import uuid4

from sourced.ml.transformers import Engine, ContentExtractor, ContentDeserializer, \
    Content2Ids, HeadFiles
from sourced.ml.utils import create_engine


def repos2ids_entry(args):
    log = logging.getLogger("repos2ids")
    engine = create_engine("repos2ids-%s" % uuid4(), **args.__dict__)

    pipeline = Engine(engine) \
        .link(HeadFiles()) \
        .link(ContentExtractor()) \
        .link(ContentDeserializer()) \
        .link(Content2Ids(args))
    ids = pipeline.execute()
    
    log.info("Writing %s", args.output)
    with gzip.open(args.output, "a") as g:
        for pair in ids.collect():
            g.write(str.encode(",".join(pair) + "\n"))
