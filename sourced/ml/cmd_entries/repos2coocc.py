import logging
from uuid import uuid4

from sourced.ml.extractors import IdentifiersBagExtractor
from sourced.ml.models import OrderedDocumentFrequencies, DocumentFrequencies
from sourced.ml.transformers import Engine, HeadFiles, UastExtractor, Cacher, UastDeserializer, \
    CooccConstructor, CooccModelSaver, Uast2DocFreq
from sourced.ml.utils import create_engine, EngineConstants


def repos2coocc_entry(args):
    log = logging.getLogger("repos2coocc")
    engine = create_engine("repos2coocc-%s" % uuid4(), **args.__dict__)
    id_extractor = IdentifiersBagExtractor(docfreq_threshold=args.min_docfreq,
                                           split_stem=args.split_stem)
    document_column_name = EngineConstants.Columns.RepositoryId
    df_transformer = Uast2DocFreq([id_extractor], document_column_name)

    pipeline = Engine(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \

    df_rdd = pipeline \
        .link(df_transformer) \
        .execute()\
        .filter(lambda x: x.value >= args.min_docfreq)
    df = df_rdd.collectAsMap()
    df_model = OrderedDocumentFrequencies() if args.ordered else DocumentFrequencies()
    df_model.construct(df_transformer.ndocs, df)
    log.info("Writing document frequency model to %s...", args.docfreq)
    df_model.save(args.docfreq)

    tokens = list(df.keys())
    token2index = df_rdd.context.broadcast({token: i for i, token in enumerate(tokens)})
    pipeline = pipeline \
        .link(CooccConstructor(token2index=token2index,
                               token_parser=id_extractor.id2bag.token_parser,
                               namespace=id_extractor.NAMESPACE)) \
        .link(CooccModelSaver(args.output, tokens, df_model))
    pipeline.execute()
