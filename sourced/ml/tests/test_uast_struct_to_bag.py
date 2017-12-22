import unittest

from bblfsh import BblfshClient

from sourced.ml.algorithms import UastRandomWalk2Bag, UastSeq2Bag
from sourced.ml.tests.models import SOURCE_PY


class Uast2RandomWalk2BagTest(unittest.TestCase):
    def setUp(self):
        self.uast_random_walk2bag = UastRandomWalk2Bag(seq_len=[2, 3])
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_uast_to_bag(self):
        bag = self.uast_random_walk2bag(self.uast)
        self.assertTrue(len(bag) > 0, "Expected size of bag should be > 0")

    def test_equivalence_prepare_starting_nodes(self):
        starting_nodes_old = self.prepare_starting_nodes(self.uast)
        starting_nodes = self.uast_random_walk2bag.uast2walks.prepare_starting_nodes(self.uast)
        self.assertEqual(len(starting_nodes_old), len(starting_nodes))

        def structure(tree):
            from collections import Counter
            return set(Counter(len(node.children) for node in tree))

        self.assertEqual(structure(starting_nodes_old), structure(starting_nodes))

    def prepare_starting_nodes(self, uast):
        starting_nodes = []
        self._prepare_starting_nodes(uast, None, starting_nodes)

        return starting_nodes

    def _prepare_starting_nodes(self, root, parent, starting_nodes):
        node = self.uast_random_walk2bag.uast2walks._extract_node(node=root, parent=parent)
        starting_nodes.append(node)

        for ch in root.children:
            node.children.append(self._prepare_starting_nodes(
                ch, parent=node, starting_nodes=starting_nodes))


class UastSeq2BagTest(unittest.TestCase):
    def setUp(self):
        self.uast_seq2bag = UastSeq2Bag(seq_len=[2, 3])
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_uast_to_bag(self):
        bag = self.uast_seq2bag(self.uast)
        self.assertTrue(len(bag) > 0, "Expected size of bag should be > 0")


if __name__ == "__main__":
    unittest.main()
