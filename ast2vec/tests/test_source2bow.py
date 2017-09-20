import argparse
import asdf
import os
import tempfile
import unittest

from ast2vec.bow import BOW
from ast2vec.model2.uast2bow import uast2bow_entry
import ast2vec.tests.models as paths


class Source2DocFreqTests(unittest.TestCase):
    def test_all(self):
        with tempfile.TemporaryDirectory(prefix="ast2vec-test-source2bow-") as tmpdir:
            args = argparse.Namespace(
                processes=2, input=paths.DATA_DIR_SOURCE, output=tmpdir,
                filter="**/source_*.asdf", vocabulary_size=500,
                docfreq=os.path.join(os.path.dirname(__file__), paths.DOCFREQ),
                overwrite_existing=True, prune_df=1)
            uast2bow_entry(args)
            for n, file in enumerate(os.listdir(tmpdir)):
                bow = BOW().load(os.path.join(tmpdir, file))
                self.assertGreater(bow._matrix.getnnz(), 0)
                self.assertEqual(len(bow.repos), 1)
            self.assertEqual(n, 3)

    def test_overwrite_existing(self):
        with tempfile.TemporaryDirectory(prefix="ast2vec-test-source2bow-") as tmpdir:
            args = argparse.Namespace(
                processes=2, input=paths.DATA_DIR_SOURCE, output=tmpdir,
                filter="**/source_*.asdf", vocabulary_size=500,
                docfreq=os.path.join(os.path.dirname(__file__), paths.DOCFREQ),
                overwrite_existing=False, prune_df=1)
            uast2bow_entry(args)
            data1 = [asdf.open(os.path.join(tmpdir, file))
                     for n, file in enumerate(os.listdir(tmpdir))]
            uast2bow_entry(args)
            data2 = [asdf.open(os.path.join(tmpdir, file))
                     for n, file in enumerate(os.listdir(tmpdir))]
            self.assertEqual([d.tree["meta"]["created_at"] for d in data1],
                             [d.tree["meta"]["created_at"] for d in data2])
            args.overwrite_existing = True
            uast2bow_entry(args)
            data3 = [asdf.open(os.path.join(tmpdir, file))
                     for n, file in enumerate(os.listdir(tmpdir))]
            self.assertNotEqual([d.tree["meta"]["created_at"] for d in data1],
                                [d.tree["meta"]["created_at"] for d in data3])


if __name__ == "__main__":
    unittest.main()
