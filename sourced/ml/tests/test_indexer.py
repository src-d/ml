import unittest

from pyspark import Row

from sourced.ml.transformers import Indexer
from sourced.ml.utils import create_spark


class IndexerTests(unittest.TestCase):
    def setUp(self):
        data = [Row(to_index='to_index1', value=1),
                Row(to_index='to_index1', value=2),
                Row(to_index='to_index2', value=3),
                Row(to_index='to_index3', value=4),
                Row(to_index='to_index4', value=4),
                Row(to_index='to_index5', value=5),
                ]
        self.data = data
        self.sc = create_spark("test")
        self.data_rdd = self.sc.sparkContext \
            .parallelize(range(len(data))) \
            .map(lambda x: data[x])

    def test_call(self):
        indexer = Indexer("to_index")
        res = indexer(self.data_rdd)
        data_reverse = res \
            .map(lambda x: Row(to_index=indexer.values[x.to_index], value=x.value)) \
            .collect()
        self.assertEqual(self.data, data_reverse)


if __name__ == "__main__":
    unittest.main()
