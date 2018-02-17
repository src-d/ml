from typing import Iterable, Dict

import numpy

from modelforge import register_model, merge_strings, split_strings
from sourced.ml.models import DocumentFrequencies


@register_model
class OrderedDocumentFrequencies(DocumentFrequencies):
    """
    Compatible with the original DocumentFrequencies. This model maintains the determinitic
    sequence of the tokens.
    """
    NAME = "ordered_docfreq"

    def construct(self, docs: int, tokfreqs: Iterable[Dict[str, int]]):
        super().construct(docs, tokfreqs)
        self._log.info("Ordering the keys...")
        keys = list(self._df)
        keys.sort()
        self._order = {k: i for i, k in enumerate(keys)}
        return self

    @property
    def order(self):
        return self._order

    def _load_tree(self, tree):
        tokens = split_strings(tree["tokens"])
        super()._load_tree(tree, tokens)
        self._log.info("Mapping the keys order...")
        self._order = {k: i for i, k in enumerate(tokens)}

    def _generate_tree(self):
        tokens = [None] * len(self)
        freqs = numpy.zeros(len(self), dtype=numpy.float32)
        for k, i in self._order.items():
            tokens[i] = k
            freqs[i] = self._df[k]
        return {"docs": self.docs, "tokens": merge_strings(tokens), "freqs": freqs}

    @staticmethod
    def maybe(enabled: bool):
        """
        Materializes either OrderedDocumentFrequencies or regular DocumentFrequencies
        depending on the specified boolean flag (true for Ordered).
        :param enabled: If true, :class:`OrderedDocumentFrequencies` instance is returned, \
                        otherwise :class:`DocumentFrequencies`.
        """
        return OrderedDocumentFrequencies() if enabled else DocumentFrequencies()
