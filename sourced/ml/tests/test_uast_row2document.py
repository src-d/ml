import unittest

from pyspark import Row

from sourced.ml.transformers import UastRow2Document


class UastRow2DocumentTest(unittest.TestCase):
    def test_documentize(self):
        r2d = UastRow2Document()
        row = Row(repository_id="1", path="2", blob_id="3", uast="4")
        row2 = r2d.documentize(row)
        row2_correct = Row(document='1//2@3', uast='4')
        self.assertEqual(row2, row2_correct)


if __name__ == "__main__":
    unittest.main()
