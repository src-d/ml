import shutil
import tempfile
import unittest

from pyspark.sql import Row

from sourced.ml.utils.spark import create_spark
from sourced.ml.transformers.basic import ParquetSaver, ParquetLoader, Collector, First, \
     Identity, FieldsSelector, Repartitioner
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

    def test_parquet_loader(self):
        # load parquet and check number of rows
        loader = ParquetLoader(session=self.spark, paths=PARQUET_DIR)
        data = loader.execute()
        self.assertEqual(data.count(), 6)

    def test_identity(self):
        # load parquet
        loader = ParquetLoader(session=self.spark, paths=PARQUET_DIR)
        data = loader.execute()
        # check that identity returns the same RDD
        data_identity = Identity()(data)
        self.assertEqual(data_identity.count(), 6)
        self.assertEqual(data_identity, data)

    def test_parquet_saver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = tmpdir
        try:
            # load and save data
            rows = [("Alice", 1)]
            df = self.spark.createDataFrame(rows, ["name", "age"])
            ParquetSaver(dirname + "/")(df.rdd)

            # read saved data and check it
            data = ParquetLoader(session=self.spark, paths=dirname).execute()
            self.assertEqual(data.count(), 1)
        finally:
            shutil.rmtree(dirname)

    def test_collector(self):
        data = ParquetLoader(session=self.spark, paths=PARQUET_DIR).link(Collector()) \
            .execute()
        self.assertEqual(len(data), 6)

    def test_first(self):
        row = ParquetLoader(session=self.spark, paths=PARQUET_DIR).link(First()) \
            .execute()
        self.assertTrue(isinstance(row, Row))

    def test_field_selector(self):
        rows = [("Alice", 1)]
        df = self.spark.createDataFrame(rows, ["name", "age"])
        # select field "name"
        row = FieldsSelector(fields=["name"])(df.rdd).first()
        self.assertFalse(hasattr(row, "age"))
        self.assertTrue(hasattr(row, "name"))
        # select field "age"
        row = FieldsSelector(fields=["age"])(df.rdd).first()
        self.assertTrue(hasattr(row, "age"))
        self.assertFalse(hasattr(row, "name"))
        # select field "name" and "age"
        row = FieldsSelector(fields=["name", "age"])(df.rdd).first()
        self.assertTrue(hasattr(row, "age"))
        self.assertTrue(hasattr(row, "name"))


if __name__ == '__main__':
    unittest.main()
