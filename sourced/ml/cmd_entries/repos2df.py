import logging
from uuid import uuid4

from sourced.ml.extractors import create_extractors_from_args
from sourced.ml.models import OrderedDocumentFrequencies
from sourced.ml.transformers import Ignition, UastExtractor, UastDeserializer, \
    BagFeatures2DocFreq, HeadFiles, Uast2BagFeatures, Counter, Cacher
from sourced.ml.utils import create_engine, EngineConstants


def repos2df_entry(args):
    log = logging.getLogger("repos2df")
    engine = create_engine("repos2df-%s" % uuid4(), **args.__dict__)
    extractors = create_extractors_from_args(args)

    df = Ignition(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(Cacher.maybe(args.persist))
    log.info("Extracting UASTs...")
    ndocs = df.link(Counter()).execute()
    log.info("Number of documents: %d", ndocs)
    df = df \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractors, EngineConstants.Columns.RepositoryId)) \
        .link(BagFeatures2DocFreq()) \
        .execute()

    log.info("Writing %s", args.docfreq)
    OrderedDocumentFrequencies().construct(ndocs, df).save(args.docfreq)
