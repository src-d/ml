import gzip
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

    pipeline = Engine(engine) \
        .link(HeadFiles()) \
        .link(ContentExtractor()) \
        .link(ContentDeserializer()) \
        .link(Content2Ids(args, documents_column_name))
    ids = pipeline.execute()

    log.info("Writing %s", args.output)
    with gzip.open(args.output, "w") as g:
        columns_names = ["token", "token_split"]
        if args.idfreq:
            columns_names.extend(["num_repos", "num_files", "num_occ"])
        g.write(str.encode(",".join(columns_names).upper() + "\n"))
        for row in ids.collect():
            row_dict = row.asDict()
            row_list = []
            for col in columns_names:
                row_list.append(str(row_dict[col]))
            g.write(str.encode(",".join(row_list) + "\n"))
