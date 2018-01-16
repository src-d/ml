from collections import defaultdict

from sourced.ml.algorithms import Uast2ChildrenCounts, Uast2InttypesAndQuantizedChildren
from sourced.ml.extractors import BagsExtractor, register_extractor


@register_extractor
class ChildrenQuantizationExtractor(BagsExtractor):
    NAME = "children_quant"
    NAMESPACE = "cq."

    def __init__(self, docfreq_threshold=None, **kwargs):
        super().__init__(docfreq_threshold=docfreq_threshold, **kwargs)
        self.uast2bag = Uast2ChildrenCounts()
        self.docfreq = defaultdict(lambda: defaultdict(int))

    def extract(self, uast):
        return []

    def uast_to_bag(self, uast):
        return self.uast_to_bag(uast)

    def apply_docfreq(self, key, value):
        internal_type, children = key.split("_")
        self.docfreq[internal_type][int(children)] = value

    def finalize(self):
        pass


@register_extractor
class ChildrenBagExtractor(BagsExtractor):
    """
    Converts a UAST to the bag of pairs (internal type, quantized number of children).
    """
    NAME = "children"
    NAMESPACE = "c."
    DEPENDS = ChildrenQuantizationExtractor,

    def __init__(self, docfreq_threshold=None,  **kwargs):
        super().__init__(docfreq_threshold=docfreq_threshold, **kwargs)
        self.uast2bag = Uast2InttypesAndQuantizedChildren(**uast2bag_kwargs)

    def uast_to_bag(self, uast):
        return self.uast2bag(uast)
