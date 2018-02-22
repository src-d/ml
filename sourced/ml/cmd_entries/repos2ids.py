import logging
from typing import NamedTuple
from uuid import uuid4

from sourced.ml.transformers import Ignition, \
    Content2Ids, ContentExtractor, HeadFiles, Cacher
from sourced.ml.utils import create_engine, EngineConstants


def repos2ids_entry(args):
    log = logging.getLogger("repos2ids")
    engine = create_engine("repos2ids-%s" % uuid4(), **args.__dict__)
    Column = NamedTuple("Column", [("repo_id", str), ("file_id", str)])
    language_mapping = Content2Ids.build_mapping()
    column_names = Column(repo_id=EngineConstants.Columns.RepositoryId,
                          file_id=EngineConstants.Columns.Path)

    ids = Ignition(engine) \
        .link(HeadFiles()) \
        .link(ContentExtractor()) \
        .link(Content2Ids(language_mapping, column_names, args.split, args.idfreq)) \
        .execute()

    log.info("Writing %s", args.output)
    ids.toDF().toPandas().to_csv(args.output)
