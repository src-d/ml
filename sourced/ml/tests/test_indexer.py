import tempfile
import unittest

from pyspark import Row

from sourced.ml.models import DocumentFrequencies
from sourced.ml.tests import create_spark_for_test
from sourced.ml.transformers import Indexer


class IndexerTests(unittest.TestCase):
    def setUp(self):
        data = [Row(to_index="to_index%d" % i, value=i) for i in range(10)]
        self.data = data
        self.session = create_spark_for_test()
        self.data_rdd = self.session.sparkContext \
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

    def test_save_load(self):
        indexer = Indexer("to_index")
        res = indexer(self.data_rdd)
        with tempfile.NamedTemporaryFile(suffix="-index.asdf") as tmp:
            cached_index_path = tmp.name
            indexer.save_index(cached_index_path)
            docfreq = DocumentFrequencies().load(source=cached_index_path)
            document_index = {key: int(val) for (key, val) in docfreq}
            indexer = Indexer("to_index", column2id=document_index)
            self.assertEqual(res.collect(), indexer(self.data_rdd).collect())


if __name__ == "__main__":
    unittest.main()
