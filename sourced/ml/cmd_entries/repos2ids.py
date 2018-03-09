import logging
from typing import NamedTuple
from uuid import uuid4

from sourced.ml.transformers import Ignition, Content2Ids, ContentExtractor, \
    ContentProcess, HeadFiles, Cacher
from sourced.ml.utils import create_engine


def repos2ids_entry(args):
    log = logging.getLogger("repos2ids")
    engine = create_engine("repos2ids-%s" % uuid4(), **args.__dict__)

    ids = Ignition(engine) \
        .link(HeadFiles()) \
        .link(ContentExtractor()) \
        .link(ContentProcess(args.split)) \
        .link(Cacher.maybe(args.persist)) \
        .link(Content2Ids(args.idfreq)) \
        .execute()

    log.info("Writing %s", args.output)
    ids.toDF() \
        .coalesce(1) \
        .write \
        .option("header", "true") \
        .mode("overwrite") \
        .csv(args.output)
