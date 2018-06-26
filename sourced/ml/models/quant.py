from typing import Dict

from modelforge import Model, split_strings, merge_strings, register_model
import numpy


@register_model
class QuantizationLevels(Model):
    """
    This model contains quantization levels for multiple schemes (feature types).
    Every feature "class" (type, possible distinct value) corresponds to the numpy array
    with integer level borders. The size of each numpy array is (the number of levels + 1).
    """
    NAME = "quant"

    def construct(self, levels: Dict[str, Dict[str, numpy.ndarray]]):
        self._levels = levels
        return self

    @property
    def levels(self) -> Dict[str, Dict[str, numpy.ndarray]]:
        return self._levels

    def __len__(self):
        return len(self.levels)

    def _load_tree(self, tree):
        self._levels = {}
        for key, vals in tree["schemes"].items():
            classes = split_strings(vals["classes"])
            levels = vals["levels"]
            self.levels[key] = dict(zip(classes, numpy.split(levels, len(classes))))

    def _generate_tree(self):
        tree = {"schemes": {}}
        for key, vals in self.levels.items():
            tree["schemes"][key] = scheme = {}
            npartitions = len(next(iter(vals.values())))
            classes = [None for _ in range(len(vals))]
            scheme["levels"] = levels = numpy.zeros(len(vals) * npartitions, dtype=numpy.int32)
            for i, pair in enumerate(vals.items()):
                classes[i], levels[i * npartitions:(i + 1) * npartitions] = pair
            scheme["classes"] = merge_strings(classes)
        return tree

    def dump(self):
        return """Schemes: %s""" % (
            sorted((v[0], "%d@%d" % (len(v[1]), len(next(iter(v[1].values()))) - 1))
                   for v in self.levels.items()))

    def apply_quantization(self, extractors):
        for extractor in extractors:
            try:
                extractor.quantize
            except AttributeError:
                continue
            extractor.uast_to_bag.levels = self._levels[extractor.NAME]
