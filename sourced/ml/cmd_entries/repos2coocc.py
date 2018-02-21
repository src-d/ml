import logging
from uuid import uuid4

from sourced.ml.extractors import IdentifiersBagExtractor
from sourced.ml.models import OrderedDocumentFrequencies
from sourced.ml.transformers import Ignition, HeadFiles, UastExtractor, Cacher, UastDeserializer, \
    CooccConstructor, CooccModelSaver, BagFeatures2DocFreq, Uast2BagFeatures, Counter
from sourced.ml.utils import create_engine, EngineConstants


def repos2coocc_entry(args):
    log = logging.getLogger("repos2coocc")
    engine = create_engine("repos2coocc-%s" % uuid4(), **args.__dict__)
    id_extractor = IdentifiersBagExtractor(docfreq_threshold=args.min_docfreq,
                                           split_stem=args.split_stem)
    uast_extractor = Ignition(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(Cacher.maybe(args.persist))
    log.info("Extracting UASTs...")
    ndocs = uast_extractor.link(Counter()).execute()
    log.info("Number of documents: %d", ndocs)
    uast_extractor = uast_extractor.link(UastDeserializer())

    df = uast_extractor \
        .link(Uast2BagFeatures([id_extractor], EngineConstants.Columns.RepositoryId)) \
        .link(BagFeatures2DocFreq()) \
        .execute()

    log.info("Writing document frequency model to %s...", args.docfreq)
    df_model = OrderedDocumentFrequencies() \
        .construct(ndocs, df) \
        .prune(args.min_docfreq) \
        .greatest(args.vocabulary_size) \
        .save(args.docfreq)

    token2index = engine.session.sparkContext.broadcast(df_model.order)
    pipeline = uast_extractor \
        .link(CooccConstructor(token2index=token2index,
                               token_parser=id_extractor.id2bag.token_parser,
                               namespace=id_extractor.NAMESPACE)) \
        .link(CooccModelSaver(args.output, df_model))
    pipeline.execute()
