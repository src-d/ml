import unittest

from sourced.ml.utils.spark import create_spark
from sourced.ml.transformers.basic import ParquetLoader, Repartitioner
from sourced.ml.tests.models import PARQUET_DIR


class BasicTransformerTest(unittest.TestCase):
    def setUp(self):
        self.spark = create_spark("test")

    def test_repartitioner(self):
        partitions = 2
        data = ParquetLoader(session=self.spark, paths=PARQUET_DIR).execute().rdd.coalesce(1)

        #  coalesce without shuffle cannot make more partitions, only concatenate them
        #  it is a shuffle flag check.
        repartitioned_data = Repartitioner(partitions, shuffle=False)(data)
        self.assertEqual(1, repartitioned_data.getNumPartitions())

        repartitioned_data = Repartitioner(partitions, shuffle=True)(data)
        self.assertEqual(partitions, repartitioned_data.getNumPartitions())


if __name__ == '__main__':
    unittest.main()
