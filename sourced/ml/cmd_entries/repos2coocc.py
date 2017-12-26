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
    token_parser = TokenParser()

    pipeline = Engine(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(Cacher.maybe(args.persist)) \
        .link(TokenMapper(token_parser))

    tokens, tokens2index = pipeline.execute()

    pipeline = pipeline \
        .link(UastDeserializer()) \
        .link(CooccConstructor(tokens2index, token_parser)) \
        .link(CooccModelSaver(args.output, tokens))

    pipeline.execute()
