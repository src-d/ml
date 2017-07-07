import argparse
import os
import tempfile
import unittest

import asdf

from ast2vec import repo2nbow, Id2Vec, DocumentFrequencies
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
        nbow = repo2nbow(
            os.path.join(basedir, "..", ".."),
            id2vec=id2vec, df=df, linguist=tests.ENRY, timeout=600)
        self.assertIsInstance(nbow, dict)
        self.assertAlmostEqual(nbow[0], 14.234776549829204)

    def test_asdf(self):
        basedir = os.path.dirname(__file__)
        with tempfile.NamedTemporaryFile() as file:
            args = argparse.Namespace(
                id2vec=os.path.join(basedir, ID2VEC),
                df=os.path.join(basedir, DOCFREQ), linguist=tests.ENRY,
                gcs=None, output=file.name, bblfsh=None, timeout=None,
                repository=os.path.join(basedir, "..", ".."))
            repo2nbow_entry(args)
            data = asdf.open(file.name)
            self.assertIn("meta", data.tree)
            self.assertIn("nbow", data.tree)
            self.assertEqual(2, len(data.tree["meta"]["dependencies"]))


if __name__ == "__main__":
    unittest.main()
