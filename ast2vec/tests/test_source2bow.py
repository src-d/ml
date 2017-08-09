import argparse
import os
import tempfile
import unittest

from ast2vec.bow import BOW
from ast2vec.model2.source2bow import source2bow_entry
import ast2vec.tests.models as paths


class Source2DocFreqTests(unittest.TestCase):
    def test_all(self):
        with tempfile.TemporaryDirectory(prefix="ast2vec-test-source2bow-") as tmpdir:
            args = argparse.Namespace(
                processes=2, input=paths.DATA_DIR_SOURCE, output=tmpdir,
                filter="**/source_*.asdf", vocabulary_size=500,
                df=os.path.join(os.path.dirname(__file__), paths.DOCFREQ))
            source2bow_entry(args)
            for n, file in enumerate(os.listdir(tmpdir)):
                bow = BOW().load(os.path.join(tmpdir, file))
                self.assertGreater(bow._matrix.getnnz(), 0)
                self.assertEqual(len(bow.repos), 1)
            self.assertEqual(n, 3)


if __name__ == "__main__":
    unittest.main()
