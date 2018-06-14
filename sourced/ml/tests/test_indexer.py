import unittest

from pyspark import Row

from sourced.ml.tests import create_spark_for_test
from sourced.ml.transformers import Indexer


class IndexerTests(unittest.TestCase):
    def setUp(self):
        data = [Row(to_index="to_index%d" % i, value=i) for i in range(10)]
        self.data = data
        self.sc = create_spark_for_test()
        self.data_rdd = self.sc.sparkContext \
            .parallelize(range(len(data))) \
            .map(lambda x: data[x])

    def test_call(self):
        indexer = Indexer("to_index")
        res = indexer(self.data_rdd)
        values = indexer.values()
        data_reverse = res \
            .map(lambda x: Row(to_index=values[x.to_index], value=x.value)) \
            .collect()
        self.assertEqual(self.data, data_reverse)


if __name__ == "__main__":
    unittest.main()
