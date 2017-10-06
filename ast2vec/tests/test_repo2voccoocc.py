import os
import unittest

import asdf
from scipy.sparse import coo_matrix

import ast2vec.tests as tests
from ast2vec.repo2.voccoocc import Repo2VocCoocc


def validate_asdf_file(obj, filename):
    data = asdf.open(filename)
    obj.assertIn("meta", data.tree)
    obj.assertIn("matrix", data.tree)
    obj.assertEqual(data.tree["meta"]["model"], "vocabulary_co-occurrences")


class Repo2VocCooccTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_obj(self):
        basedir = os.path.dirname(__file__)
        repo2 = Repo2VocCoocc(vocabulary={
            "basedir": 0,
            "repo": 1,
            "test": 2,
        }, linguist=tests.ENRY)
        coocc = repo2.convert_repository(os.path.join(basedir, "..", ".."))
        self.assertIsInstance(coocc, coo_matrix)
        self.assertEqual(coocc.shape, (3, 3))
        self.assertEqual(coocc.getnnz(), 6)


if __name__ == "__main__":
    unittest.main()
