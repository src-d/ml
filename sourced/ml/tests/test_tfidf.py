import unittest

from pyspark import Row

from sourced.ml.tests import tfidf_data
from sourced.ml.transformers import TFIDF
from sourced.ml.utils import create_spark


class TFIDFTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark("test")

        tf = list(tfidf_data.term_freq_result[-1])
        tf_rdd = self.sc.sparkContext \
            .parallelize(range(len(tf))) \
            .map(lambda x: tf[x])

        df = list(tfidf_data.doc_freq_result[-1])
        df_rdd = self.sc.sparkContext \
            .parallelize(range(len(df))) \
            .map(lambda x: df[x])

        self.tfidf = TFIDF(tf=tf_rdd, df=df_rdd, docfreq_threshold=1)

    def test_call(self):
        result = {Row(document=0, token='test.word5', value=4.0731170968132782),
                  Row(document=0, token='test.word6', value=3.4632661925263397),
                  Row(document=1, token='test.word5', value=4.0731170968132782),
                  Row(document=1, token='test.word8', value=5.4598979633004001),
                  Row(document=1, token='test.word2', value=4.0731170968132782),
                  Row(document=1, token='test.word1', value=4.9571937336190652),
                  Row(document=0, token='test.word2', value=1.6523979112063552),
                  Row(document=0, token='test.word1', value=1.0425470069194165),
                  Row(document=1, token='test.word9', value=7.3216016464675571),
                  Row(document=1, token='test.word6', value=4.1701880276776659),
                  Row(document=0, token='test.word4', value=3.9368979424627821),
                  Row(document=1, token='test.word7', value=5.2687144272709192),
                  Row(document=0, token='test.word3', value=5.9501944585220681)}

        self.assertEqual(result, set(self.tfidf(None).collect()))


if __name__ == "__main__":
    unittest.main()
