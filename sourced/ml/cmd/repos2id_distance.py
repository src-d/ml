import logging
from uuid import uuid4

from sourced.ml.extractors import IdentifierDistance
from sourced.ml.transformers import UastDeserializer, Uast2BagFeatures, UastRow2Document, \
    CsvSaver, create_uast_source
from sourced.ml.transformers.basic import Rower
from sourced.ml.utils.engine import pipeline_graph, pause


@pause
def repos2id_distance(args):
    log = logging.getLogger("repos2roles_and_ids")
    extractor = IdentifierDistance(args.split, args.type, args.max_distance)
    session_name = "repos2roles_and_ids-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)

    start_point \
        .link(UastRow2Document()) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractor)) \
        .link(Rower(lambda x: dict(identifier1=x[0][0][0],
                                   identifier2=x[0][0][1],
                                   distance=x[1]))) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
