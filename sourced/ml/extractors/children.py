import numpy
import re
from collections import defaultdict

from sourced.ml.algorithms import Uast2NodesBag
from sourced.ml.extractors import BagsExtractor, register_extractor, get_names_from_kwargs,\
    filter_kwargs


@register_extractor
class ChildrenBagExtractor(BagsExtractor):
    """
    Converts a UAST to a bag of features that are composed of the nodes' internal type
    and their quantized number of children.

    As regards quantization performed in class: Repo2Quant
    get_children_freq() is first invoked to build the set of the number of children frequencies.
    Then quantize() is called to fill the quantization mapping thanks to
    build_partition() and process_value() methods.
    """
    NAME = "children"
    NAMESPACE = "c."
    OPTS = dict(get_names_from_kwargs(Uast2NodesBag.__init__))
    OPTS.update(BagsExtractor.OPTS)

    def __init__(self, docfreq_threshold=None, **kwargs):
        super().__init__(docfreq_threshold)
        self._log.debug("__init__ %s", kwargs)
        self.mapping = dict()
        uast2bag_kwargs = filter_kwargs(kwargs, Uast2NodesBag.__init__)
        self.uast2bag = Uast2NodesBag(**uast2bag_kwargs)

    def inspect(self, uast):
        """
        This method is overridden to update the keys in the bag of features
        after quantization.
        """
        try:
            bag, _ = self.uast_to_bag(uast)
        except RuntimeError as e:
            raise ValueError(str(uast)) from e
        for key in bag:
            yield self.NAMESPACE + self.update(key)

    def inspect_quant(self, uast):
        """
        This method is used to process UASTs to access the numbers of children
        in class: Repo2Quant.
        """
        try:
            _, all_children = self.uast_to_bag(uast)
        except RuntimeError as e:
            raise ValueError(str(uast)) from e
        for nb_children in all_children:
            yield str(nb_children)

    def update(self, key):
        """
        Substitutes in the string key the original number of children
        with its quantized value.
        """
        children_quant = self.mapping[key.split("_")[-1]]
        return re.sub(r"\d{1,}", str(children_quant), key)

    def extract(self, uast):
        """
        Converts a UAST to the weighted set.
        This method is overridden in order to update the bag of features
        after quantization of the number of children in class: Repo2WeightedSet.
        Needs merge_children() and update() methods to update the frequencies
        in the bag with the right keys.
        """
        ndocs = self.ndocs
        docfreq = self.docfreq
        log = numpy.log
        bag, _ = self.uast_to_bag(uast)
        bag = self.merge_children(bag)
        for key, val in bag.items():
            key = self.NAMESPACE + key
            try:
                yield key, log(1 + val) * log(ndocs / docfreq[key]) * self.scale
            except KeyError:
                # docfreq_threshold
                continue

    def get_children_freq(self, reduced):
        children_freq = {}
        for key, value in reduced.items():
            children_freq[key] = value
        return children_freq

    def merge_children(self, bag):
        new_bag = defaultdict(int)
        for key, val in bag.items():
            new_bag[self.update(key)] += val
        return new_bag

    def quantize(self, children_freq, nb_partitions):
        try:
            partition = self.build_partition(children_freq, nb_partitions)
            for value in set(children_freq):
                self.mapping[value] = self.process_value(int(value), partition)
        finally:
            self.nb_partitions = None

    def build_partition(self, children_freq, nb_partitions):
        """
        Builds the quantization partition P that is a vector of length nb_partitions \
        whose entries are in strictly ascending order.
        The quantization index corresponding to an input value of x is:
            0 if x <= P[0]
            m if P[m-1] < x <= P[m]
            n if P[n] <= x

        :param children_freq: distribution of the nodes's number of children \
        we want to quantize.
        :param nb_partitions: length of the partition vector.
        :return: The vector of endpoints of the partition intervals.
        """
        partition = numpy.zeros(nb_partitions)
        max_nodes_per_bin = sum(list(children_freq.values())) / nb_partitions
        values = [int(v) for v in list(children_freq)]
        values.sort()
        j = 0
        while children_freq[str(values[j])] > max_nodes_per_bin:
            partition[j] = values[j]
            j += 1
        nb_bins_left = numpy.count_nonzero(numpy.asarray(partition) == 0) - 1
        new_start = numpy.nonzero(numpy.asarray(partition[1:]) == 0)[0][0] + 1
        children_freq_sorted = list(children_freq.values())
        children_freq_sorted.sort(reverse=True)
        max_nodes_per_bin = sum(children_freq_sorted[new_start:]) / nb_bins_left
        id_val = new_start
        for i in range(new_start, nb_partitions):
            nb_nodes_cum = 0
            while (nb_nodes_cum < max_nodes_per_bin):
                nb_nodes_cum += values[id_val] * children_freq[str(values[id_val])]
                id_val += 1
            partition[i] = values[id_val]
        return partition

    def process_value(self, value, partition):
        """
        Produce quantization index and quantized output value.

        :param value: value we want to quantize.
        :return: the corresponding quantization index.
        """
        linear_values = range(len(partition))
        idx = numpy.searchsorted(partition, value, side="right")
        return linear_values[idx-1]

    def uast_to_bag(self, uast):
        return self.uast2bag(uast)
