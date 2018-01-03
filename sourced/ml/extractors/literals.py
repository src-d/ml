import codecs
import os

from sourced.ml.algorithms.uast_ids_to_bag import UastIds2Bag
from sourced.ml.extractors import BagsExtractor, register_extractor


class HashedTokenParser:
    def process_token(self, token):
        yield codecs.encode((hash(token) & 0xffffffffffffffff).to_bytes(8, "little"),
                            "hex_codec").decode()


class Literals2Bag(UastIds2Bag):
    """
    Converts a UAST to a bag-of-literals.
    """

    XPATH = "//*[@roleLiteral]"

    def __init__(self, token2index=None, token_parser=None):
        """
        :param token2index: The mapping from tokens to bag keys. If None, no mapping is performed.
        :param token_parser: Specify token parser if you want to use a custom one. \
            :class:'TokenParser' is used if it is not specified.
        """
        token_parser = HashedTokenParser() if token_parser is None else token_parser
        super().__init__(token2index, token_parser)


@register_extractor
class LiteralsBagExtractor(BagsExtractor):
    NAME = "lit"
    NAMESPACE = "l."
    OPTS = BagsExtractor.OPTS.copy()

    def __init__(self, docfreq_threshold=None, **kwargs):
        super().__init__(docfreq_threshold, **kwargs)
        self.id2bag = Literals2Bag(None, HashedTokenParser())

    def uast_to_bag(self, uast):
        if os.getenv("PYTHONHASHSEED", "random") == "random":
            raise RuntimeError("PYTHONHASHSEED must be set")
        return self.id2bag(uast)
