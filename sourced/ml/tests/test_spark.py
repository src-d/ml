import unittest

from sourced.ml.utils.spark import get_spark_memory_config, create_spark
from sourced.ml.tests.models import PARQUET_DIR


class SparkTests(unittest.TestCase):
    def test_assemble_spark_config(self):
        config = get_spark_memory_config()
        self.assertEqual(len(config), 0)
        config = get_spark_memory_config(memory="1,2,3")
        self.assertEqual(config, ("spark.executor.memory=1",
                                  "spark.driver.memory=2",
                                  "spark.driver.maxResultSize=3"))
        self.assertRaises(ValueError, get_spark_memory_config, memory="1,2,3,4")

    def test_create_spark(self):
        spark = create_spark("test_1")
        self.assertEqual(spark.read.parquet(PARQUET_DIR).count(), 6)
        spark.stop()
        spark = create_spark("test_2", dep_zip=False, config=["spark.rpc.retry.wait=10s"])
        self.assertEqual(spark.conf.get("spark.rpc.retry.wait"), "10s")
        spark.stop()


if __name__ == '__main__':
    unittest.main()
