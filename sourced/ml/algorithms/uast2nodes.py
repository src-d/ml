from collections import defaultdict

from sourced.ml.algorithms.uast_ids_to_bag import Uast2BagBase


class Uast2NodesBag(Uast2BagBase):
    """
    Converts a UAST to a bag of features that are node specific.
    The features are pairs of internal type and number of children)
    """
    def uast2nodes(self, root):
        """
        :param uast: The UAST root node.
        :generate: The nodes which compose the UAST.
        """
        stack = [root]
        while stack:
            child = stack.pop()
            stack.extend(child.children)
            yield child

    def node2key(self, node):
        """
        :param node: a node of UAST
        :return: The string joining 2 features of node : \
        Its internal type and its quantized number of children \
        str format is required for wmhash.Bags.Extractor.
        """
        return "%s_%s" % (node.internal_type, len(node.children))

    def __call__(self, uast):
        """
        Converts a UAST to a bag of features. The weights are feature frequencies.
        :param uast: The UAST root node.
        :return: bag of features and the list of the number of children encountered.
        """
        bag = defaultdict(int)
        all_children = []
        for node in self.uast2nodes(uast):
            bag[self.node2key(node)] += 1
            all_children.append(len(node.children))
        return bag, all_children
