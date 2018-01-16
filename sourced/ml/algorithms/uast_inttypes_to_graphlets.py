from collections import defaultdict

from sourced.ml.algorithms.uast_ids_to_bag import Uast2BagBase
from sourced.ml.algorithms.uast_struct_to_bag import Node


class Uast2GraphletBag(Uast2BagBase):
    """
    Converts a UAST to a bag of graphlets.
    The graphlet of a UAST node is composed from the node itself, its parent and its children.
    Each node is represented by the internal role string.
    """
    @staticmethod
    def _extract_node(node, parent):
        return Node(parent=parent, internal_type=node.internal_type)

    def uast2graphlets(self, uast):
        """
        :param uast: The UAST root node.
        :generate: The nodes which compose the UAST.
            :class: 'Node' is used to access the nodes of the graphlets.
        """
        root = self._extract_node(uast, None)
        stack = [(root, uast)]
        while stack:
            parent, parent_uast = stack.pop()
            children_nodes = [self._extract_node(child, parent) for child in parent_uast.children]
            parent.children = children_nodes
            stack.extend(zip(children_nodes, parent_uast.children))
            yield parent

    def node2key(self, node):
        """
        Builds the string joining internal types of all the nodes
        in the node's graphlet in the following order:
        parent_node_child1_child2_child3. The children are sorted by alphabetic order.
        str format is required for BagsExtractor.

        :param node: a node of UAST
        :return: The string key of node
        """
        try:
            parent_type = node.parent.internal_type
        except AttributeError:
            parent_type = None
        key = [parent_type, node.internal_type]
        key.extend(sorted(ch.internal_type for ch in node.children))
        return "_".join(map(str, key))

    def __call__(self, uast):
        """
        Converts a UAST to a weighed bag of graphlets. The weights are graphlets frequencies.
        :param uast: The UAST root node.
        :return: bag of graphlets.
        """
        bag = defaultdict(int)
        for node in self.uast2graphlets(uast):
            bag[self.node2key(node)] += 1
        return bag
