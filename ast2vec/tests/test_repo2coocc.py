import argparse
from collections import namedtuple
import os
import tempfile
import unittest

import asdf
from scipy.sparse import coo_matrix

from ast2vec import Repo2Coocc, Repo2CooccTransformer
from ast2vec.bblfsh_roles import SIMPLE_IDENTIFIER
import ast2vec.tests as tests
from ast2vec.repo2.coocc import repo2coocc_entry


def validate_asdf_file(obj, filename):
    data = asdf.open(filename)
    obj.assertIn("meta", data.tree)
    obj.assertIn("matrix", data.tree)
    obj.assertIn("tokens", data.tree)
    obj.assertEqual(data.tree["meta"]["model"], "co-occurrences")


class Repo2CooccTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_obj(self):
        basedir = os.path.dirname(__file__)
        repo2 = Repo2Coocc(linguist=tests.ENRY)
        coocc = repo2.convert_repository(os.path.join(basedir, "..", ".."))
        self.assertIsInstance(coocc, tuple)
        self.assertEqual(len(coocc), 2)
        self.assertIn("document", coocc[0])
        self.assertIsInstance(coocc[1], coo_matrix)
        self.assertEqual(coocc[1].shape, (len(coocc[0]),) * 2)
        self.assertGreater(coocc[1].getnnz(), 20000)

    def test_asdf(self):
        basedir = os.path.dirname(__file__)
        with tempfile.NamedTemporaryFile() as file:
            args = argparse.Namespace(
                linguist=tests.ENRY, output=file.name,
                repository=os.path.join(basedir, "..", ".."),
                bblfsh_endpoint=None, timeout=None)
            repo2coocc_entry(args)
            validate_asdf_file(self, file.name)

    def test_zero_tokens(self):
        def skip_uast(root, word2ind, dok_mat):
            pass

        repo2 = Repo2Coocc(linguist=tests.ENRY)
        repo2._traverse_uast = skip_uast
        basedir = os.path.dirname(__file__)
        coocc = repo2.convert_repository(os.path.join(basedir, "..", ".."))
        self.assertEqual(coocc[0], [])
        self.assertEqual(coocc[1].shape, (0, 0))
        self.assertEqual(coocc[1].nnz, 0)

    def test_extract_ids(self):
        Node = namedtuple("Node", ["roles", "token", "children"])
        node1 = Node([SIMPLE_IDENTIFIER], 1, [])
        node2 = Node([], 2, [])
        node3 = Node([SIMPLE_IDENTIFIER], 3, [node1, node2])
        node4 = Node([SIMPLE_IDENTIFIER], 4, [])
        root = Node([], 5, [node3, node4])
        repo2 = Repo2Coocc(linguist=tests.ENRY)
        self.assertEqual(list(repo2._extract_ids(root)), [4, 3, 1])

    def test_linguist(self):
        # If this test fails, check execution permissions for provided paths.
        with self.assertRaises(FileNotFoundError):
            Repo2Coocc(linguist="xxx")
        with self.assertRaises(FileNotFoundError):
            Repo2Coocc(linguist=__file__)


class Repo2CooccTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_transform(self):
        basedir = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as tmpdir:
            r2cc = Repo2CooccTransformer(
                linguist=tests.ENRY)
            r2cc.transform(repos=basedir, output=tmpdir)

            # check that output file exists
            outfile = r2cc.prepare_filename(basedir, tmpdir)
            self.assertEqual(os.path.exists(outfile), 1)

            validate_asdf_file(self, outfile)

    def test_empty(self):
        basedir = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as tmpdir:
            r2cc = Repo2CooccTransformer(
                linguist=tests.ENRY)
            r2cc.transform(repos=os.path.join(basedir, "coocc"), output=tmpdir)
            self.assertFalse(os.listdir(tmpdir))


if __name__ == "__main__":
    unittest.main()
