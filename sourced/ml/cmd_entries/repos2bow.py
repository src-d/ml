import logging
from uuid import uuid4

from scipy import sparse

from sourced.ml.extractors import __extractors__
from sourced.ml.models import BOW, OrderedDocumentFrequencies, QuantizationLevels
from sourced.ml.transformers import Engine, UastExtractor, UastDeserializer, Uast2Quant, \
    Uast2DocFreq, Uast2TermFreq, HeadFiles, TFIDF, Cacher, Indexer
from sourced.ml.utils import create_engine, EngineConstants


def repos2bow_entry(args):
    log = logging.getLogger("repos2bow")
    engine = create_engine("repos2bow-%s" % uuid4(), **args.__dict__)
    document_column_name = EngineConstants.Columns.RepositoryId
    extractors = [__extractors__[s](
        args.min_docfreq, log_level=args.log_level,
        **__extractors__[s].get_kwargs_fromcmdline(args))
        for s in args.feature]

    uast_extractor = Engine(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer())
    quant = Uast2Quant(extractors)
    uast_extractor.link(quant).execute()
    if quant.levels:
        log.info("Writing quantization levels to %s", args.quant)
        QuantizationLevels().construct(quant.levels).save(args.quant)
    df = uast_extractor.link(Uast2DocFreq(extractors, document_column_name)).execute()
    log.info("Calculating the raw document frequencies...")
    df_collected = df.collectAsMap()
    log.info("Done")
    tf = uast_extractor.link(Uast2TermFreq(extractors, document_column_name)).execute()
    document_indexer = Indexer(TFIDF.Columns.document)
    token_indexer = Indexer(TFIDF.Columns.token)
    documents_id, tokens_id, values = zip(
        *TFIDF(tf=tf, df=df)
        .link(Cacher.maybe(args.persist))
        .link(document_indexer)
        .link(token_indexer)
        .execute().collect())
    ndocs = len(document_indexer.value_to_index)
    ntokens = len(token_indexer.value_to_index)

    log.info("Writing docfreq to %s", args.docfreq)
    model_df = OrderedDocumentFrequencies.maybe(args.ordered) \
        .construct(ndocs, df_collected) \
        .save(args.docfreq)

    log.info("Writing BOW to %s", args.bow)
    matrix = sparse.csr_matrix((values, (documents_id, tokens_id)), shape=(ndocs, ntokens))
    BOW() \
        .construct(document_indexer.values, matrix, token_indexer.values) \
        .save(args.bow, deps=(model_df,))
