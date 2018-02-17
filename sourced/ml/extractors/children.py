import logging
from typing import Iterable, Tuple

from sourced.ml.algorithms import Uast2QuantizedChildren
from sourced.ml.extractors import BagsExtractor, register_extractor, filter_kwargs, \
    get_names_from_kwargs


@register_extractor
class ChildrenBagExtractor(BagsExtractor):
    """
    Converts a UAST to the bag of pairs (internal type, quantized number of children).
    """
    NAME = "children"
    NAMESPACE = "c."
    OPTS = dict(get_names_from_kwargs(Uast2QuantizedChildren.__init__))

    def __init__(self, docfreq_threshold=None, **kwargs):
        original_kwargs = kwargs
        uast2bag_kwargs = filter_kwargs(kwargs, Uast2QuantizedChildren.__init__)
        for k in uast2bag_kwargs:
            kwargs.pop(k)
        super().__init__(docfreq_threshold, **kwargs)
        self._log.debug("__init__ %s", original_kwargs)
        self.uast_to_bag = Uast2QuantizedChildren(**uast2bag_kwargs)

    @property
    def npartitions(self):
        return self.uast_to_bag.npartitions

    @property
    def levels(self):
        return self.uast_to_bag.levels

    def extract(self, uast):
        if not self.uast_to_bag.levels:
            # bypass NAMESPACE
            gen = self.uast_to_bag(uast).items()
        else:
            gen = super().extract(uast)
        for key, val in gen:
            yield key, val

    def quantize(self, frequencies: Iterable[Tuple[str, Iterable[Tuple[int, int]]]]):
        self.uast_to_bag.quantize(frequencies)
        if self._log.isEnabledFor(logging.DEBUG):
            for k, v in self.uast_to_bag.levels.items():
                self._log.debug("%s\n%s", k, v)
