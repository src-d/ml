import logging
from uuid import uuid4

from sourced.ml.extractors import __extractors__
from sourced.ml.models import OrderedDocumentFrequencies, DocumentFrequencies
from sourced.ml.transformers import Engine, UastExtractor, Cacher, UastDeserializer, \
    BagsBatcher, Repo2DocFreq, Repo2WeightedSet, HeadFiles
from sourced.ml.utils import create_engine


def repos2df_entry(args):
    log = logging.getLogger("repos2df")
    if len(args.feature) > 1 and not args.ordered:
        raise ValueError("If you want to save document frequency model for several features"
                         "please add --ordered flag. If you want to have default document frequency"
                         " model specify only one feature.")
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
    if args.ordered:
        model = OrderedDocumentFrequencies()
        model.construct(extractors[0].ndocs, [e.docfreq for e in extractors])
    else:
        model = DocumentFrequencies()
        model.construct(extractors[0].ndocs, extractors[0].docfreq)
    log.info("Writing %s", args.docfreq)
    model.save(args.docfreq)
