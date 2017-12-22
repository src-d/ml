import logging
from uuid import uuid4

from sourced.ml.algorithms import TokenParser
from sourced.ml.transformers import Engine, HeadFiles, UastExtractor, Cacher, UastDeserializer, \
    CooccConstructor, CooccModelSaver
from sourced.ml.transformers.token_mapper import TokenMapper
from sourced.ml.utils import create_engine


def repos2coocc_entry(args):
    log = logging.getLogger("repos2cooc")
    engine = create_engine("repos2cooc-%s" % uuid4(), **args.__dict__)

    pipeline = Engine(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages))
    pipeline = Cacher.maybe(pipeline, args.persist)

    token_parser = TokenParser()
    tokens, tokens2index = pipeline.link(TokenMapper(token_parser)).execute()

    pipeline = pipeline \
        .link(UastDeserializer()) \
        .link(CooccConstructor(tokens2index, token_parser)) \
        .link(CooccModelSaver(args.output, tokens))

    pipeline.execute()
