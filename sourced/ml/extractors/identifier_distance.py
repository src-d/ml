from typing import Iterable, Tuple

import bblfsh

from sourced.ml.extractors.bags_extractor import BagsExtractor
from sourced.ml.algorithms import NoopTokenParser, Uast2IdTreeDistance, Uast2IdLineDistance
from sourced.ml.algorithms.uast_id_distance import Uast2IdDistance


class IdentifierDistance(BagsExtractor):
    """
    Extractor wrapper for Uast2IdTreeDistance and Uast2IdLineDistance algorithm.
    Note that this is an unusual BagsExtractor since it returns iterable instead of bag.

    The class did not wrap with @register_extractor because it does not produce bags as others do.
    So nobody outside code will see it or use it directly.
    For the same reason we a free to override NAMESPACE, NAME, OPTS fields with any value we want.

    TODO(zurk): Split BagsExtractor into two clases: Extractor and BagsExtractor(Extractor),
    re-inherit this class from Extractor, delete explanations from docstring.
    """
    NAMESPACE = ""
    NAME = "Identifier distance"
    OPTS = {}
    DEFAULT_MAX_DISTANCE = Uast2IdDistance.DEFAULT_MAX_DISTANCE

    class DistanceType:
        Tree = "tree"
        Line = "line"
        All = {Tree, Line}

        @staticmethod
        def resolve(type):
            if type == IdentifierDistance.DistanceType.Line:
                return Uast2IdLineDistance
            if type == IdentifierDistance.DistanceType.Tree:
                return Uast2IdTreeDistance
            raise ValueError("Unknown distance type: %s" % type)

    def __init__(self, split_stem=False, type="tree", max_distance=DEFAULT_MAX_DISTANCE, **kwargs):
        super().__init__(**kwargs)
        Uast2IdDistance = self.DistanceType.resolve(type)
        self.uast2id_distance = Uast2IdDistance(
            token_parser=NoopTokenParser() if not split_stem else None,
            max_distance=max_distance)

    def extract(self, uast: bblfsh.Node) -> Iterable[Tuple[str, str, int]]:
        yield from self.uast2id_distance(uast)
