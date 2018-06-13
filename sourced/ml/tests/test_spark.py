import unittest

from sourced.ml.utils.spark import assemble_spark_config, create_spark
from sourced.ml.tests.models import PARQUET_DIR


class SparkTests(unittest.TestCase):
    @unittest.skip("Until assemble_spark_config alter Spark default values.")
    def test_assemble_spark_config(self):
        config = assemble_spark_config()
        self.assertEqual(len(config), 0)
        config = assemble_spark_config(memory="1,2,3")
        self.assertListEqual(config, ["spark.executor.memory=1",
                                      "spark.driver.memory=2",
                                      "spark.driver.maxResultSize=3"])

    def test_create_spark(self):
        spark = create_spark("test_1")
        self.assertEqual(spark.read.parquet(PARQUET_DIR).count(), 6)
        spark.stop()
        spark = create_spark("test_2", dep_zip=False, config=["spark.rpc.retry.wait=10s"])
        self.assertEqual(spark.conf.get("spark.rpc.retry.wait"), "10s")
        spark.stop()


if __name__ == '__main__':
    unittest.main()
