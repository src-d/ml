import unittest

from pyspark import Row

from sourced.ml.tests.test_uast2docfreq import TestExtractor
from sourced.ml.tests import tfidf_data
from sourced.ml.transformers import Uast2TermFreq
from sourced.ml.utils import create_spark


class Uast2TermFreqTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark("test")
        self.uast2df = Uast2TermFreq([TestExtractor()], "document")

    def test_call(self):
        for data, result in zip(tfidf_data.datasets, tfidf_data.term_freq_result):
            rdd = self.sc.sparkContext \
                .parallelize(range(len(data["uast"]))) \
                .map(lambda x: Row(uast=data["uast"][x], document=data["document"][x]))
            self.assertEqual(result, set(self.uast2df(rdd).collect()))


if __name__ == "__main__":
    unittest.main()
