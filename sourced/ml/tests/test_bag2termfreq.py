import unittest

from sourced.ml.tests import create_spark_for_test, tfidf_data
from sourced.ml.transformers import BagFeatures2TermFreq


class Uast2TermFreqTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark_for_test()
        self.bag2tf = BagFeatures2TermFreq()

    def test_call(self):
        tf = self.bag2tf(self.sc.sparkContext.parallelize(
            [((i["t"], i["d"]), i["v"]) for i in tfidf_data.dataset])) \
            .map(lambda r: {k[0]: v for k, v in r.asDict().items()}) \
            .collect()
        self.assertEqual({tfidf_data.readonly(i) for i in tf}, tfidf_data.term_freq_result)


if __name__ == "__main__":
    unittest.main()
