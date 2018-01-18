import logging
from uuid import uuid4

from sourced.ml.extractors import __extractors__
from sourced.ml.models import OrderedDocumentFrequencies, DocumentFrequencies
from sourced.ml.transformers import Engine, UastExtractor, Cacher, UastDeserializer, \
    BagsBatcher, Uast2DocFreq, Uast2TermFreq, HeadFiles
from sourced.ml.utils import create_engine, EngineConstants


def repos2df_entry(args):
    log = logging.getLogger("repos2df")
    engine = create_engine("repos2df-%s" % uuid4(), **args.__dict__)
    extractors = [__extractors__[s](
        args.min_docfreq, **__extractors__[s].get_kwargs_fromcmdline(args))
        for s in args.feature]
    document_column_name = EngineConstants.Columns.RepositoryId
    df_transformer = Uast2DocFreq(extractors, document_column_name)

    pipeline = Engine(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(UastDeserializer()) \
        .link(df_transformer)
    df = pipeline.execute()

    model = OrderedDocumentFrequencies() if args.ordered else DocumentFrequencies()
    model.construct(df_transformer.ndocs, df.collectAsMap())
    log.info("Writing %s", args.docfreq)
    model.save(args.docfreq)
