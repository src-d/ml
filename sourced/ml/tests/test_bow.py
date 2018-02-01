import os
import unittest

import numpy

import sourced.ml.tests.models as paths
from sourced.ml.models import BOW, DocumentFrequencies


class BOWTests(unittest.TestCase):
    def setUp(self):
        self.model = BOW().load(source=os.path.join(os.path.dirname(__file__), paths.BOW))

    def test_getitem(self):
        repo_name, indices, weights = self.model[0]
        self.assertEqual(repo_name, "repo1")
        self.assertIsInstance(indices, numpy.ndarray)
        self.assertIsInstance(weights, numpy.ndarray)
        self.assertEqual(indices.shape, weights.shape)
        self.assertEqual(indices.shape, (3,))

    def test_iter(self):
        pumped = list(self.model)
        self.assertEqual(len(pumped), 5)
        self.assertEqual(pumped, list(range(5)))

    def test_len(self):
        self.assertEqual(len(self.model), 5)

    def test_repository_index_by_name(self):
        self.assertEqual(
            self.model.documents_index_by_name("repo1"), 0)

    def test_tokens(self):
        self.assertEqual(self.model.tokens[0], "i.")


if __name__ == "__main__":
    unittest.main()
