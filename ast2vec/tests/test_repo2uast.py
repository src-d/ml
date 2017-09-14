import argparse
import os
import tempfile
import unittest

import asdf
from ast2vec.bblfsh_roles import Node
from modelforge import split_strings

from ast2vec import Repo2UASTModelTransformer, Repo2UASTModel
from ast2vec.repo2.uast import repo2uast_entry
import ast2vec.tests as tests


def validate_asdf_file(obj, filename):
    data = asdf.open(filename)
    obj.assertIn("meta", data.tree)
    obj.assertIn("filenames", data.tree)
    obj.assertIn("uasts", data.tree)
    obj.assertIn("repository", data.tree)
    obj.assertEqual(data.tree["meta"]["model"], "uast")
    Node.FromString(split_strings(data.tree["uasts"])[0])
    obj.assertEqual(0, len(data.tree["meta"]["dependencies"]))
    obj.assertEqual(data.tree["meta"]["model"], "uast")


class Repo2UASTModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_obj(self):
        basedir = os.path.dirname(__file__)
        repo2 = Repo2UASTModel(linguist=tests.ENRY)
        prox = repo2.convert_repository(os.path.join(basedir, "..", ".."))
        self.assertIsInstance(prox, tuple)
        self.assertEqual(len(prox), 2)
        self.assertIn("ast2vec/__init__.py", prox[0])

    def test_asdf(self):
        basedir = os.path.dirname(__file__)
        with tempfile.NamedTemporaryFile() as file:
            args = argparse.Namespace(
                linguist=tests.ENRY, output=file.name,
                repository=os.path.join(basedir, "..", ".."),
                bblfsh_endpoint=None, timeout=None)
            repo2uast_entry(args)
            validate_asdf_file(self, file.name)

    def test_linguist(self):
        with self.assertRaises(FileNotFoundError):
            Repo2UASTModel(linguist="xxx")
        with self.assertRaises(FileNotFoundError):
            Repo2UASTModel(linguist=__file__)


class Repo2UASTModelTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_transform(self):
        basedir = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo2 = Repo2UASTModelTransformer(
                linguist=tests.ENRY)
            repo2.transform(repos=basedir, output=tmpdir)

            # check that output file exists
            outfile = repo2.prepare_filename(basedir, tmpdir)
            self.assertEqual(os.path.exists(outfile), 1)

            validate_asdf_file(self, outfile)

    def test_empty(self):
        basedir = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo2 = Repo2UASTModelTransformer(
                linguist=tests.ENRY)
            repo2.transform(repos=os.path.join(basedir, "coocc"), output=tmpdir)
            self.assertFalse(os.listdir(tmpdir))

if __name__ == "__main__":
    unittest.main()
