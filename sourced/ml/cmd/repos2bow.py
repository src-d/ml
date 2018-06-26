import logging
from uuid import uuid4

from sourced.ml.extractors import create_extractors_from_args
from sourced.ml.transformers import UastDeserializer, BagFeatures2TermFreq, Uast2BagFeatures, \
    HeadFiles, TFIDF, Cacher, Indexer, UastRow2Document, BOWWriter, Moder, create_uast_source, \
    Repartitioner
from sourced.ml.utils.engine import pipeline_graph, pause
from sourced.ml.utils.docfreq import create_or_load_ordered_df
from sourced.ml.utils.quant import create_or_apply_quant


@pause
def repos2bow_template(args, select=HeadFiles, cache_hook=None, save_hook=None):
    log = logging.getLogger("repos2bow")
    extractors = create_extractors_from_args(args)
    session_name = "repos2bow-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name, select=select)
    uast_extractor = start_point.link(Moder(args.mode)) \
        .link(Repartitioner.maybe(args.partitions, args.shuffle)) \
        .link(Cacher.maybe(args.persist))
    if cache_hook is not None:
        uast_extractor.link(cache_hook()).execute()
    # We link UastRow2Document after Cacher here because cache_hook() may want to have all possible
    # Row items.
    uast_extractor = uast_extractor.link(UastRow2Document())
    log.info("Extracting UASTs and indexing documents...")
    document_indexer = Indexer(Uast2BagFeatures.Columns.document,
                               cached_index_path=args.cached_index_path)
    if args.cached_index_path is None:
        uast_extractor.link(document_indexer).execute()
    ndocs = len(document_indexer)
    log.info("Number of documents: %d", ndocs)
    uast_extractor = uast_extractor.link(UastDeserializer())
    if args.quant:
        create_or_apply_quant(args.quant, extractors, uast_extractor)
    uast_extractor = uast_extractor \
        .link(Uast2BagFeatures(extractors))
    df_model = create_or_load_ordered_df(args, ndocs, uast_extractor)
    bags_writer = uast_extractor \
        .link(BagFeatures2TermFreq()) \
        .link(TFIDF(df_model)) \
        .link(document_indexer) \
        .link(Indexer(Uast2BagFeatures.Columns.token, df_model.order))
    if save_hook is not None:
        bags_writer = bags_writer \
            .link(Repartitioner.maybe(args.partitions, args.shuffle, multiplier=10)) \
            .link(save_hook())
    bags_writer.link(BOWWriter(document_indexer, df_model, args.bow, args.batch)) \
        .execute()
    pipeline_graph(args, log, root)


def repos2bow(args):
    return repos2bow_template(args)


def repos2bow_index_template(args, select=HeadFiles):
    log = logging.getLogger("repos2bow_index")
    extractors = create_extractors_from_args(args)
    session_name = "repos2bow_index_features-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name, select=select)
    uast_extractor = start_point.link(Moder(args.mode)) \
        .link(Repartitioner.maybe(args.partitions, args.shuffle)) \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist))
    log.info("Extracting UASTs and indexing documents...")
    document_indexer = Indexer(Uast2BagFeatures.Columns.document)
    uast_extractor.link(document_indexer).execute()
    document_indexer.save_index(args.cached_index_path)
    ndocs = len(document_indexer)
    log.info("Number of documents: %d", ndocs)
    uast_extractor = uast_extractor.link(UastDeserializer())
    if args.quant:
        create_or_apply_quant(args.quant, extractors, uast_extractor)
    if args.docfreq_out:
        create_or_load_ordered_df(args, ndocs, uast_extractor.link(Uast2BagFeatures(extractors)))
    pipeline_graph(args, log, root)


def repos2bow_index(args):
    return repos2bow_index_template(args)
