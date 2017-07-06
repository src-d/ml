import os
import unittest

from ast2vec import repo2nbow, Id2Vec, DocumentFrequencies
import ast2vec.tests as tests
from ast2vec.tests.models import ID2VEC, DOCFREQ


class Repo2NBOWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_ast2vec(self):
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
        self.assertAlmostEqual(nbow[0], 13.79585695137316)


if __name__ == "__main__":
    unittest.main()
