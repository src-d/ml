import argparse
import os
import tempfile
import unittest

import asdf
from scipy.sparse import coo_matrix

from ast2vec import Repo2Coocc
from ast2vec.repo2coocc import repo2coocc_entry
import ast2vec.tests as tests


class Repo2CooccTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_obj(self):
        basedir = os.path.dirname(__file__)
        repo2 = Repo2Coocc(linguist=tests.ENRY, timeout=600)
        coocc = repo2.convert_repository(os.path.join(basedir, "..", ".."))
        self.assertIsInstance(coocc, tuple)
        self.assertEqual(len(coocc), 2)
        self.assertIn("document", coocc[0])
        self.assertIsInstance(coocc[1], coo_matrix)
        self.assertEqual(coocc[1].shape, (len(coocc[0]),) * 2)

    def test_asdf(self):
        basedir = os.path.dirname(__file__)
        with tempfile.NamedTemporaryFile() as file:
            args = argparse.Namespace(
                linguist=tests.ENRY, output=file.name, bblfsh_endpoint=None,
                timeout=None, repository=os.path.join(basedir, "..", ".."))
            repo2coocc_entry(args)
            data = asdf.open(file.name)
            self.assertIn("meta", data.tree)
            self.assertIn("matrix", data.tree)
            self.assertIn("tokens", data.tree)
            self.assertEqual(data.tree["meta"]["model"], "co-occurrences")


if __name__ == "__main__":
    unittest.main()
