import logging
from uuid import uuid4

from sourced.ml.extractors import IdSequenceExtractor
from sourced.ml.transformers import (
    create_uast_source, CsvSaver, Uast2BagFeatures, UastDeserializer, UastRow2Document)
from sourced.ml.transformers.basic import Rower
from sourced.ml.utils.engine import pause, pipeline_graph


@pause
def repos2id_sequence(args):
    log = logging.getLogger("repos2id_distance")
    extractor = IdSequenceExtractor(args.split)
    session_name = "repos2roles_and_ids-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)
    if not args.skip_docname:
        mapper = Rower(lambda x: {"document": x[0][1],
                                  "identifiers": x[0][0]})
    else:
        mapper = Rower(lambda x: {"identifiers": x[0][0]})
    start_point \
        .link(UastRow2Document()) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractor)) \
        .link(mapper) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
