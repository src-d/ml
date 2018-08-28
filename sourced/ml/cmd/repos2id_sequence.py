import logging
from uuid import uuid4

from sourced.ml.extractors import IdSequenceExtractor
from sourced.ml.transformers import UastDeserializer, Uast2BagFeatures, UastRow2Document, \
    CsvSaver, create_uast_source
from sourced.ml.transformers.basic import Rower
from sourced.ml.utils.engine import pipeline_graph, pause


@pause
def repos2id_sequence(args):
    log = logging.getLogger("repos2id_distance")
    extractor = IdSequenceExtractor(args.split)
    session_name = "repos2roles_and_ids-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)
    if not args.skip_docname:
        mapper = Rower(lambda x: dict(document=x[0][1],
                                      identifiers=x[0][0]))
    else:
        mapper = Rower(lambda x: dict(identifiers=x[0][0]))
    start_point \
        .link(UastRow2Document()) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractor)) \
        .link(mapper) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
