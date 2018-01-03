import numpy
import re
from collections import defaultdict

from sourced.ml.algorithms import UastNodes2Bag
from sourced.ml.extractors import BagsExtractor, register_extractor, get_names_from_kwargs,\
    filter_kwargs


@register_extractor
class TypeChildrenBagExtractor(BagsExtractor):
    NAME = "typechildren"
    NAMESPACE = "tc."
    OPTS = dict(get_names_from_kwargs(UastNodes2Bag.__init__))
    OPTS = BagsExtractor.OPTS.copy()

    def __init__(self, docfreq_threshold=None, nb_partitions=10, **kwargs):
        super().__init__(docfreq_threshold, nb_partitions)
        self._log.debug("__init__ %s", kwargs)
        self.mapping = dict()
        uast2bag_kwargs = filter_kwargs(kwargs, UastNodes2Bag.__init__)
        self.uast2bag = UastNodes2Bag(**uast2bag_kwargs)

    def inspect(self, uast):
        try:
            bag, all_children = self.uast_to_bag(uast)
        except RuntimeError as e:
            raise ValueError(str(uast)) from e
        if self.nb_partitions:
            for nb_children in all_children:
                yield str(nb_children)
        else:
            for key in bag:
                yield self.NAMESPACE + self.quantize(key)

    def quantize(self, key):
        children_quant = self.mapping[key.split("_")[-1]]
        return re.sub(r"\d{1,}", str(children_quant), key)

    def extract(self, uast):
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
        NDOCS_KEY = -1, 0
        for (i, key), value in reduced.items():
            if (i, key) != NDOCS_KEY:
                children_freq[key] = value
        return children_freq

    def merge_children(self, bag):
        new_bag = defaultdict(int)
        for key, val in bag.items():
            new_bag[self.quantize(key)] += val
        return new_bag

    def build_quantization(self, children_freq):
        try:
            partition = self.build_partition(children_freq)
            for value in set(children_freq):
                self.mapping[value] = self.process_value(int(value), partition)
        finally:
            self.nb_partitions = None

    def build_partition(self, children_freq):
        """
        Builds the partition of the quantization.
        It is a list of increasing integers equally partitioning the distribution of values.

        :param children_freq: distribution of values we want to quantize.
        :param nb_partitions: number of intervals in the quantization partition.
        :return:
        """
        partition = numpy.zeros(self.nb_partitions)
        max_nodes_per_bin = sum(list(children_freq.values())) / self.nb_partitions
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
        try:
            for i in range(new_start, self.nb_partitions):
                nb_nodes_cum = 0
                while (nb_nodes_cum < max_nodes_per_bin):
                    nb_nodes_cum += values[id_val] * children_freq[str(values[id_val])]
                    id_val += 1
                partition[i] = values[id_val]
        finally:
            partition[-1] = values[-1] + 1
        return partition

    def process_value(self, value, partition):
        """
        Processes value according to the quantization algorithm. \
        Behaves like a stair function whose set of permissible outputs are values in partition \
        projected to linear values.

        :param value: value we want to quantize.
        :return:
        """
        linear_values = range(len(partition))
        idx = numpy.searchsorted(partition, value, side="right")
        return linear_values[idx-1]

    def uast_to_bag(self, uast):
        return self.uast2bag(uast)
