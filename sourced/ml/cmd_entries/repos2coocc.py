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

    pipeline = Engine(engine, explain=args.explain)
    pipeline = pipeline.link(HeadFiles())
    pipeline = pipeline.link(UastExtractor(languages=args.languages))
    pipeline = Cacher.maybe(pipeline, args.persist)

    token_parser = TokenParser()
    tokens, tokens2index = pipeline.link(TokenMapper(token_parser)).execute()

    uasts = pipeline.link(UastDeserializer())
    tokens_matrix = uasts.link(CooccConstructor(tokens2index, token_parser))
    save_model = tokens_matrix.link(CooccModelSaver(args.output, tokens))
    save_model.execute()
