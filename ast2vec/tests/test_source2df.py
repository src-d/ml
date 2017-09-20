import argparse
import tempfile
import unittest

from ast2vec.df import DocumentFrequencies
from ast2vec.model2.uast2df import uast2df_entry
import ast2vec.tests.models as paths


class Source2DocFreqTests(unittest.TestCase):
    def test_all(self):
        with tempfile.NamedTemporaryFile(prefix="ast2vec-test-source2df-", suffix=".asdf") as tmpf:
            args = argparse.Namespace(
                processes=2, input=paths.DATA_DIR_SOURCE, output=tmpf.name, tmpdir=None,
                filter="**/source_*.asdf")
            uast2df_entry(args)
            merged = DocumentFrequencies().load(tmpf.name)
        self.assertEqual(len(merged), 335)


if __name__ == "__main__":
    unittest.main()
