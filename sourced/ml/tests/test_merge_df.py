import os
import tempfile
import unittest

from sourced.ml.models import DocumentFrequencies
from sourced.ml.models.model_converters.merge_df import MergeDocFreq


class Model2BaseTests(unittest.TestCase):
    def setUp(self):
        self.model1 = DocumentFrequencies().construct(3, {"one": 1, "two": 2,  "three": 3})
        self.model2 = DocumentFrequencies().construct(3, {"four": 4, "three": 3, "five": 5})
        self.merge_df = MergeDocFreq(min_docfreq=1, vocabulary_size=100)
        self.merge_result = {"one": 1, "two": 2,  "three": 6, "four": 4, "five": 5}

    def test_convert_model(self):
        self.merge_df.convert_model(self.model1)
        self.assertEqual(self.merge_df._docs, 3)
        self.assertEqual(self.merge_df._df, self.model1._df)
        self.merge_df.convert_model(self.model2)
        self.assertEqual(self.merge_df._docs, 6)
        self.assertEqual(self.merge_df._df, self.merge_result)

    def test_finalize(self):
        self.merge_df.convert_model(self.model1)
        self.merge_df.convert_model(self.model2)
        with tempfile.TemporaryDirectory(prefix="merge-df-") as tmpdir:
            dest = os.path.join(tmpdir, "df.asdf")
            self.merge_df.finalize(0, dest)
            df = DocumentFrequencies().load(dest)
            self.assertEqual(df.docs, 6)
            self.assertEqual(df._df, self.merge_result)

    def test_save_path(self):
        self.assertEqual(self.merge_df._save_path(0, "df.asdf"), "df.asdf")
        self.assertEqual(self.merge_df._save_path(0, "df"), os.path.join("df", "docfreq_0.asdf"))


if __name__ == '__main__':
    unittest.main()
