import logging
from uuid import uuid4

from sourced.ml.extractors import RoleIdsExtractor
from sourced.ml.transformers import UastDeserializer, Uast2Features, UastRow2Document, \
    CsvSaver, create_uast_source, Rower
from sourced.ml.utils.engine import pipeline_graph, pause


@pause
def repos2roles_and_ids(args):
    log = logging.getLogger("repos2roles_and_ids")
    session_name = "repos2roles_and_ids-%s" % uuid4()
    extractor = RoleIdsExtractor()
    root, start_point = create_uast_source(args, session_name)

    start_point \
        .link(UastRow2Document()) \
        .link(UastDeserializer()) \
        .link(Uast2Features(extractor)) \
        .link(Rower(lambda x: dict(identifier=x["roleids"][0], role=x["roleids"][1]))) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
