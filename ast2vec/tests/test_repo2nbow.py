import argparse
import os
import tempfile
import unittest

import asdf

from ast2vec import Repo2nBOW, Id2Vec, DocumentFrequencies
from ast2vec.repo2nbow import repo2nbow_entry
import ast2vec.tests as tests
from ast2vec.tests.models import ID2VEC, DOCFREQ


class Repo2NBOWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_obj(self):
        basedir = os.path.dirname(__file__)
        id2vec = Id2Vec(os.path.join(basedir, ID2VEC))
        df = DocumentFrequencies(os.path.join(basedir, DOCFREQ))
        df._df["document"] = 10
        id2vec.tokens[0] = "document"
        id2vec._token2index["document"] = 0
        repo2nbow = Repo2nBOW(
            id2vec=id2vec, docfreq=df, linguist=tests.ENRY, timeout=600)
        nbow = repo2nbow.convert_repository(os.path.join(basedir, "..", ".."))
        self.assertIsInstance(nbow, dict)
        self.assertAlmostEqual(nbow[0], 14.635478748983617)

    def test_asdf(self):
        basedir = os.path.dirname(__file__)
        with tempfile.NamedTemporaryFile() as file:
            args = argparse.Namespace(
                id2vec=os.path.join(basedir, ID2VEC),
                docfreq=os.path.join(basedir, DOCFREQ), linguist=tests.ENRY,
                gcs_bucket=None, output=file.name, bblfsh_endpoint=None, timeout=None,
                repository=os.path.join(basedir, "..", ".."))
            repo2nbow_entry(args)
            self.assertTrue(os.path.isfile(file.name))
            data = asdf.open(file.name)
            self.assertIn("meta", data.tree)
            self.assertIn("nbow", data.tree)
            self.assertEqual(2, len(data.tree["meta"]["dependencies"]))


if __name__ == "__main__":
    unittest.main()
