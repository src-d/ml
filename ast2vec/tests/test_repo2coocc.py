import os
import unittest

from scipy.sparse import coo_matrix

from ast2vec import repo2coocc
import ast2vec.tests as tests


class Repo2CooccTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_ast2vec(self):
        basedir = os.path.dirname(__file__)
        coocc = repo2coocc(
            os.path.join(basedir, "..", ".."),
            linguist=tests.ENRY, timeout=600)
        self.assertIsInstance(coocc, tuple)
        self.assertEqual(len(coocc), 2)
        self.assertIn("document", coocc[0])
        self.assertIsInstance(coocc[1], coo_matrix)
        self.assertEqual(coocc[1].shape, (len(coocc[0]),) * 2)


if __name__ == "__main__":
    unittest.main()
