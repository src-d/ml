import logging
from uuid import uuid4

from scipy import sparse

from sourced.ml.extractors import __extractors__
from sourced.ml.models import BOW, OrderedDocumentFrequencies, DocumentFrequencies
from sourced.ml.transformers import Engine, UastExtractor, UastDeserializer, Uast2DocFreq, \
    Uast2TermFreq, HeadFiles, TFIDF, Cacher, Indexer
from sourced.ml.utils import create_engine, EngineConstants


def repos2bow_entry(args):
    log = logging.getLogger("repos2bow")
    engine = create_engine("repos2bow-%s" % uuid4(), **args.__dict__)
    document_column_name = EngineConstants.Columns.RepositoryId
    extractors = [__extractors__[s](
        args.min_docfreq, **__extractors__[s].get_kwargs_fromcmdline(args))
        for s in args.feature]

    uast_pipeline = Engine(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer())
    df_pipeline = uast_pipeline.link(Uast2DocFreq(extractors, document_column_name))
    df = df_pipeline.execute()
    tf_pipeline = uast_pipeline.link(Uast2TermFreq(extractors, document_column_name))
    tf = tf_pipeline.execute()

    document_idexer = Indexer(TFIDF.Columns.document)
    token_indexer = Indexer(TFIDF.Columns.token)
    tfidf = TFIDF(tf=tf, df=df) \
        .link(Cacher.maybe(args.persist)) \
        .link(document_idexer) \
        .link(token_indexer) \
        .execute()

    documents_id, tokens_id, values = zip(*tfidf.collect())

    ndocs = len(document_idexer.value_to_index)
    ntokens = len(token_indexer.value_to_index)
    model_df = OrderedDocumentFrequencies() if args.ordered else DocumentFrequencies()
    model_df.construct(ndocs, df.collectAsMap())
    log.info("Writing %s", args.docfreq)
    model_df.save(args.docfreq)

    matrix = sparse.csc_matrix((values, (documents_id, tokens_id)), shape=(ntokens, ndocs))
    model_bow = BOW().construct(document_idexer.values, matrix, token_indexer.values)
    log.info("Writing BOW model %s", args.bow)
    model_bow.save(args.bow, deps=(model_df,))
