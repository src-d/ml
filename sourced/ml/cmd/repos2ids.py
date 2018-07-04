import logging
from uuid import uuid4

from sourced.ml.transformers import ContentToIdentifiers, create_file_source, \
    IdentifiersToDataset, CsvSaver, Repartitioner
from sourced.ml.utils.engine import pipeline_graph, pause


@pause
def repos2ids(args):
    log = logging.getLogger("repos2ids")
    session_name = "repos2ids-%s" % uuid4()

    root, start_point = create_file_source(args, session_name)
    start_point \
        .link(Repartitioner(args.partitions, args.shuffle)) \
        .link(ContentToIdentifiers(args.split)) \
        .link(IdentifiersToDataset(args.idfreq)) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
