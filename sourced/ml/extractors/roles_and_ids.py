from typing import Iterable, Tuple

import bblfsh

from sourced.ml.algorithms.uast_to_role_id_pairs import Uast2RoleIdPairs
from sourced.ml.extractors.bags_extractor import BagsExtractor
from sourced.ml.algorithms import NoopTokenParser


class RolesAndIdsExtractor(BagsExtractor):
    """
    Extractor wrapper for Uast2RoleIdPairs algorithm.
    Note that this is an unusual BagsExtractor since it returns iterable instead of bag.

    The class did not wrap with @register_extractor because it does not produce bags as others do.
    So nobody outside code will see it or use it directly.
    For the same reason we a free to override NAMESPACE, NAME, OPTS fields with any value we want.

    TODO(zurk): Split BagsExtractor into two clases: Extractor and BagsExtractor(Extractor),
    re-inherit this class from Extractor, delete explanations from docstring.
    """
    NAMESPACE = ""
    NAME = "roles and ids"
    OPTS = {}

    def __init__(self, split_stem=False, **kwargs):
        super().__init__(**kwargs)
        self.uast2role_id_pair = Uast2RoleIdPairs(
            None, NoopTokenParser() if not split_stem else None)

    def extract(self, uast: bblfsh.Node) -> Iterable[Tuple[str, str]]:
        yield from self.uast2role_id_pair(uast)
