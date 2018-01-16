from typing import Iterable, Dict

import numpy

from sourced.ml.algorithms.uast_to_bag import Uast2BagThroughSingleScan


class Uast2ChildrenCounts(Uast2BagThroughSingleScan):
    """
    Converts a UAST to a bag of children counts.
    """

    def node2key(self, node):
        """
        :param node: a node in UAST.
        :return: The string which consists of the internal type of the node and its number of
        children.
        str format is required for wmhash.Bags.Extractor.
        """
        return "%s_%d" % (node.internal_type, len(node.children))

    @staticmethod
    def quantize(children_freq: Dict[int, int], partitions: int):
        """
        Builds the quantization partition P that is a vector of length nb_partitions \
        whose entries are in strictly ascending order.
        Quantization of x is defined as:
            0 if x <= P[0]
            m if P[m-1] < x <= P[m]
            n if P[n] <= x

        :param children_freq: distribution of the number of children.
        :param partitions: number of quantization levels.
        :return: The array with quantization levels.
        """
        levels = numpy.zeros(partitions + 1, dtype=numpy.int32)
        max_nodes_per_bin = sum(children_freq.values()) / partitions
        values = sorted(children_freq)
        levels[0] = values[0]
        accum = children_freq[values[0]]
        i = 1
        for v in values[1:]:
            f = children_freq[v]
            accum += f
            if accum > max_nodes_per_bin:
                accum = f
                if i < len(levels):
                    levels[i] = v
                    i += 1
        if i < len(levels):
            levels[i:] = values[-1]
        else:
            levels[-1] = values[-1]
        return levels


class Uast2InttypesAndQuantizedChildren(Uast2BagThroughSingleScan):
    """
    Converts a UAST to a bag of pairs (internal type, quantized number of children).
    """
    def __init__(self, inttype_seq: Iterable[str], quant_maps: Iterable[numpy.ndarray]):
        """
        Initializes a new instance of Uast2InttypesAndQuantizedChildren class.
        :param inttype_seq: The sequence of internal type values.
        :param quant_maps: The quantization levels for each internal type. \
                           The sequence matches inttype_seq.
        """
        self.quant_maps = dict(zip(inttype_seq, quant_maps))

    def node2key(self, node):
        qm = self.quant_maps[node.internal_type]
        quant_index = numpy.searchsorted(qm, len(node.children), side="right") - 1
        return "%s_%d" % (node.internal_type, quant_index)
