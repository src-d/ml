from typing import Iterable, Dict, List

import numpy

from modelforge import register_model, merge_strings, split_strings
from sourced.ml.models import DocumentFrequencies


@register_model
class OrderedDocumentFrequencies(DocumentFrequencies):
    """
    Compatible with the original DocumentFrequencies. This model maintains the determinitic
    sequence of the tokens.
    """
    # NAME is the same

    def construct(self, docs: int, tokfreqs: Iterable[Dict[str, int]]):
        super().construct(docs, tokfreqs)
        self._log.info("Ordering the keys...")
        keys = sorted(self._df)
        self._order = {k: i for i, k in enumerate(keys)}
        return self

    @property
    def order(self) -> Dict[str, int]:
        return self._order

    def tokens(self) -> List[str]:
        arr = [None for _ in range(len(self))]
        for k, v in self.order.items():
            arr[v] = k
        return arr

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

    def prune(self, threshold: int) -> "OrderedDocumentFrequencies":
        pruned = super().prune(threshold)
        if pruned is not self:
            self._log.info("Recovering the order...")
            pruned._order = {k: i for i, k in enumerate(sorted(pruned._df))}
        return pruned

    def greatest(self, max_size: int) -> "OrderedDocumentFrequencies":
        pruned = super().greatest(max_size)
        if pruned is not self:
            self._log.info("Recovering the order...")
            pruned._order = {k: i for i, k in enumerate(sorted(pruned._df))}
        return pruned
