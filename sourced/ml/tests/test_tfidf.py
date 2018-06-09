import unittest

from pyspark import Row

from sourced.ml.algorithms import log_tf_log_idf
from sourced.ml.models import DocumentFrequencies
from sourced.ml.tests import tfidf_data, create_spark_for_test
from sourced.ml.transformers import TFIDF


class TFIDFTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark_for_test()

        df = DocumentFrequencies().construct(10, {str(i): i for i in range(1, 5)})
        self.tfidf = TFIDF(df=df)

        class Columns:
            """
            Stores column names for return value.
            """
            token = "t"
            document = "d"
            value = "v"

        self.tfidf.Columns = Columns

    def test_call(self):
        baseline = {
            Row(d=dict(i)["d"], t=dict(i)["t"],
                v=log_tf_log_idf(dict(i)["v"], int(dict(i)["t"]), self.tfidf.df.docs))
            for i in tfidf_data.term_freq_result
        }

        result = self.tfidf(
            self.sc.sparkContext
                .parallelize(tfidf_data.term_freq_result)
                .map(lambda x: Row(**dict(x)))).collect()
        self.assertEqual(set(result), baseline)


if __name__ == "__main__":
    unittest.main()
