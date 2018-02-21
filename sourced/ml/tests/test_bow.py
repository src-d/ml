from io import BytesIO
import unittest

import numpy

import sourced.ml.tests.models as paths
from sourced.ml.models import BOW


class BOWTests(unittest.TestCase):
    def setUp(self):
        self.model = BOW().load(source=paths.BOW)

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

    def test_tokens(self):
        self.assertEqual(self.model.tokens[0], "i.")

    def test_write(self):
        buffer = BytesIO()
        self.model.save(buffer)
        buffer.seek(0)
        new_model = BOW().load(buffer)
        self.assertEqual((self.model.matrix != new_model.matrix).nnz, 0)
        self.assertEqual(self.model.documents, new_model.documents)
        self.assertEqual(self.model.tokens, new_model.tokens)


if __name__ == "__main__":
    unittest.main()
