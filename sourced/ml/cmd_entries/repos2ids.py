import logging
from uuid import uuid4

from sourced.ml.transformers import Engine, ContentExtractor, ContentDeserializer, \
    Content2Ids, HeadFiles, Cacher
from sourced.ml.utils import create_engine, EngineConstants


def repos2ids_entry(args):
    log = logging.getLogger("repos2ids")
    engine = create_engine("repos2ids-%s" % uuid4(), **args.__dict__)
    documents_column_name = [EngineConstants.Columns.RepositoryId,
                             EngineConstants.Columns.PathId]
    ids_transformer = Content2Ids(args, documents_column_name)

    pipeline = Engine(engine) \
        .link(HeadFiles()) \
        .link(ContentExtractor()) \
        .link(ContentDeserializer()) \
        .link(ids_transformer)
    ids = pipeline.execute()

    log.info("Writing %s", args.output)
    ids_transformer.save(ids)
