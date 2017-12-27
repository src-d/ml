import logging
from uuid import uuid4

from sourced.ml.extractors import __extractors__
from sourced.ml.models import OrderedDocumentFrequencies
from sourced.ml.transformers import Engine, UastExtractor, Cacher, UastDeserializer, \
    BagsBatcher, Repo2DocFreq, Repo2WeightedSet, HeadFiles
from sourced.ml.utils import create_engine


def repos2df_entry(args):
    log = logging.getLogger("repos2df")
    engine = create_engine("repos2df-%s" % uuid4(), **args.__dict__)
    extractors = [__extractors__[s](
        args.min_docfreq, **__extractors__[s].get_kwargs_fromcmdline(args))
        for s in args.feature]

    pipeline = Engine(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(UastDeserializer()) \
        .link(Repo2DocFreq(extractors))
    pipeline.explode()

    model = OrderedDocumentFrequencies()
    model.construct(extractors[0].ndocs, [e.docfreq for e in extractors])
    log.info("Writing %s", args.docfreq)
    model.save(args.docfreq)
