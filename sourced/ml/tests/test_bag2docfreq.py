import unittest

from sourced.ml.tests import create_spark_for_test, tfidf_data
from sourced.ml.transformers import BagFeatures2DocFreq


class Uast2DocFreqTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark_for_test()
        self.bag2df = BagFeatures2DocFreq()

    def test_call(self):
        df = self.bag2df(self.sc.sparkContext.parallelize(
            [((i["t"], i["d"]), i["v"]) for i in tfidf_data.dataset]))
        self.assertEqual(df, tfidf_data.doc_freq_result)


if __name__ == "__main__":
    unittest.main()
