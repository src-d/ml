from typing import Iterable

import bblfsh

from sourced.ml.algorithms.uast_id_distance import Uast2IdLineDistance


class Uast2IdSequence(Uast2IdLineDistance):
    """
    Converts a UAST to a sorted sequence of identifiers.
    Identifiers are sorted by position in code.
    We do not change the order if positions are not present.

    __call__ is overridden here and return list instead of bag-of-words (dist).
    """

    def __call__(self, uast: bblfsh.Node) -> str:
        """
        Converts a UAST to a sorted sequence of identifiers.
        Identifiers are sorted by position in code.
        We do not change the order if positions are not present.

        :param uast: The UAST root node.
        :return: string with a sequence of identifiers
        """
        return self.concat(id for id, pos in sorted(self._process_uast(uast), key=lambda x: x[1]))

    @staticmethod
    def concat(id_sequence: Iterable):
        return ' '.join(id_sequence)
