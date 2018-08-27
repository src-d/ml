import logging
from uuid import uuid4

from sourced.ml.extractors import create_extractors_from_args
from sourced.ml.models import OrderedDocumentFrequencies
from sourced.ml.transformers import UastDeserializer, BagFeatures2DocFreq, Uast2BagFeatures, \
    Counter, Cacher, UastRow2Document, create_uast_source
from sourced.ml.utils.engine import pipeline_graph, pause
from sourced.ml.utils.quant import create_or_apply_quant


@pause
def repos2df(args):
    log = logging.getLogger("repos2df")
    extractors = create_extractors_from_args(args)
    session_name = "repos2df-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)

    uast_extractor = start_point \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist))
    log.info("Extracting UASTs...")
    ndocs = uast_extractor.link(Counter()).execute()
    log.info("Number of documents: %d", ndocs)
    uast_extractor = uast_extractor.link(UastDeserializer())
    if args.quant:
        create_or_apply_quant(args.quant, extractors, uast_extractor)
    df = uast_extractor \
        .link(Uast2BagFeatures(*extractors)) \
        .link(BagFeatures2DocFreq()) \
        .execute()
    log.info("Writing docfreq model to %s", args.docfreq_out)
    OrderedDocumentFrequencies().construct(ndocs, df).save(args.docfreq_out)
    pipeline_graph(args, log, root)
