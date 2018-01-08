import unittest

from bblfsh import BblfshClient

from sourced.ml.algorithms import UastGraph2Bag
from sourced.ml.tests.models import SOURCE_PY


class UastGraph2BagTest(unittest.TestCase):
    def setUp(self):
        self.graphlets2bag_extractor = UastGraph2Bag()
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_uast_to_bag(self):
        bag = self.graphlets2bag_extractor(self.uast)
        self.assertTrue(len(bag) > 0, "Expected size of bag should be > 0")


if __name__ == "__main__":
    unittest.main()
