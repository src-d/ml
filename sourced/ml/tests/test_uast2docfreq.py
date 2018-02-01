import unittest

from pyspark import Row

from sourced.ml.extractors import BagsExtractor
from sourced.ml.tests import tfidf_data
from sourced.ml.transformers import Uast2DocFreq
from sourced.ml.utils import create_spark


class TestExtractor(BagsExtractor):
    NAME = "test"
    NAMESPACE = "test."

    def __init__(self):
        super().__init__(1)

    def uast_to_bag(self, uast):
        return tfidf_data.bags[uast]


class Uast2DocFreqTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark("test")
        self.uast2df = Uast2DocFreq([TestExtractor()], "document")

    def test_call(self):

        for data, result in zip(tfidf_data.datasets, tfidf_data.doc_freq_result):
            rdd = self.sc.sparkContext \
                .parallelize(range(len(data["uast"]))) \
                .map(lambda x: Row(uast=data["uast"][x], document=data["document"][x]))
            self.assertEqual(result, set(self.uast2df(rdd).collect()))


if __name__ == "__main__":
    unittest.main()
