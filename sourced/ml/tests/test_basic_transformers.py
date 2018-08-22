import csv
import os
import shutil
import tempfile
import unittest

from pyspark import StorageLevel
from pyspark.sql import Row

from sourced.ml.utils import create_engine, SparkDefault
from sourced.ml.transformers import ParquetSaver, ParquetLoader, Collector, First, \
     Identity, FieldsSelector, Repartitioner, DzhigurdaFiles, CsvSaver, Rower, \
     PartitionSelector, Sampler, Distinct, Cacher, Ignition, HeadFiles, LanguageSelector, \
     UastExtractor, UastDeserializer
from sourced.ml.tests.models import PARQUET_DIR, SIVA_DIR


class BasicTransformerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("test_with_engine", SIVA_DIR, "siva")
        cls.spark = cls.engine.session
        cls.data = ParquetLoader(session=cls.spark, paths=PARQUET_DIR).execute().rdd.coalesce(1)

    def test_repartitioner(self):
        partitions = 2

        #  coalesce without shuffle cannot make more partitions, only concatenate them
        #  it is a shuffle flag check.
        repartitioned_data = Repartitioner(partitions, shuffle=False)(self.data)
        self.assertEqual(1, repartitioned_data.getNumPartitions())

        repartitioned_data = Repartitioner(partitions, shuffle=True)(self.data)
        self.assertEqual(partitions, repartitioned_data.getNumPartitions())

        repartitioned_data = Repartitioner.maybe(partitions, shuffle=True, multiplier=2)(self.data)
        self.assertEqual(partitions * 2, repartitioned_data.getNumPartitions())

        repartitioned_data = Repartitioner.maybe(None, shuffle=False)(self.data)
        self.assertEqual(1, repartitioned_data.getNumPartitions())

        repartitioned_data = Repartitioner.maybe(partitions, keymap=lambda x: x[0])(self.data)
        self.assertEqual(repartitioned_data.count(), 6)

    def test_partition_selector(self):
        partitioned_data = PartitionSelector(partition_index=0)(self.data)
        self.assertEqual(partitioned_data.count(), 6)

    def test_sampler(self):
        sampled_data = Sampler()(self.data)
        self.assertEqual(sampled_data.count(), 2)

    def test_parquet_loader(self):
        # load parquet and check number of rows
        loader = ParquetLoader(session=self.spark, paths=(PARQUET_DIR, PARQUET_DIR))
        data = loader.execute()
        self.assertEqual(data.count(), 6 * 2)

        loader = ParquetLoader(session=self.spark, paths=PARQUET_DIR)
        data = loader.execute()
        self.assertEqual(data.count(), 6)

        self.assertEqual(loader.paths, PARQUET_DIR)
        self.assertNotIn("session", loader.__getstate__())

        try:
            loader = ParquetLoader(session=self.spark, paths=None)
            data = loader.execute()
        except ValueError:
            pass

    def test_rower(self):
        rows = [("get_user", 3)]
        df = self.spark.createDataFrame(rows, ["identifier", "frequency"])
        data = Rower(lambda x: dict(identifier=x[0], frequency=x[1]))(df.rdd)
        self.assertEqual(data.count(), 1)
        self.assertEqual(data.collect()[0].identifier, "get_user")

    def test_dzhigurda(self):
        self.assertEqual(DzhigurdaFiles(0)(self.engine).count(), 325)
        self.assertEqual(DzhigurdaFiles(10)(self.engine).count(), 3490)
        self.assertEqual(DzhigurdaFiles(-1)(self.engine).count(), 27745)

    def test_identity(self):
        # load parquet
        loader = ParquetLoader(session=self.spark, paths=PARQUET_DIR)
        data = loader.execute()
        # check that identity returns the same RDD
        data_identity = Identity()(data)
        self.assertEqual(data_identity.count(), 6)
        self.assertEqual(data_identity, data)

    def test_distinct(self):
        rows = [("foo_bar", 3), ("baz", 5), ("foo_bar", 3)]
        df = self.spark.createDataFrame(rows, ["identifier", "frequency"])
        self.assertEqual(Distinct()(df).count(), df.count() - 1)

    def test_cacher(self):
        persistence = SparkDefault.STORAGE_LEVEL
        cacher = Cacher(persistence)
        cached_data = cacher(self.data)
        self.assertTrue(cached_data.is_cached)
        self.assertEqual(cacher.persistence, getattr(StorageLevel, persistence))
        self.assertIn("head", cacher.__getstate__())

        cacher = Cacher.maybe(persistence=None)
        uncached_data = cacher(self.data)
        self.assertEqual(uncached_data, self.data)

        cacher = Cacher.maybe(persistence)
        cached_data = cacher(self.data)
        self.assertTrue(cached_data.is_cached)

        cached_data = Cacher.maybe(persistence)(self.data)
        self.assertFalse(cached_data.unpersist().is_cached)

    def test_ignition(self):
        start_point = Ignition(self.engine)
        columns = start_point(self).repositories.columns
        self.assertNotIn("engine", start_point.__getstate__())
        self.assertEqual(columns, ["id", "urls", "is_fork", "repository_path"])

    def test_head_files(self):
        df = HeadFiles()(self.engine)
        df_as_dict = df.first().asDict()
        self.assertIn("commit_hash", df_as_dict.keys())
        self.assertIn("path", df_as_dict.keys())
        self.assertIn("content", df_as_dict.keys())
        self.assertIn("reference_name", df_as_dict.keys())

    def test_uast_extractor(self):
        df = HeadFiles()(self.engine)
        df_uast = UastExtractor()(df)
        self.assertIn("uast", df_uast.columns)

    def test_uast_deserializer(self):
        df = HeadFiles()(self.engine)
        df_uast = UastExtractor()(df)
        uasts_empty = list(UastDeserializer().deserialize_uast(df.first()))
        uasts = list(UastDeserializer().deserialize_uast(df_uast.first()))
        self.assertTrue(len(uasts_empty) == 0)
        self.assertTrue(len(uasts) > 0)

    def test_csv_saver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = tmpdir

        # load and save data
        rows = [("Alice", 1)]
        df = self.spark.createDataFrame(rows, ["name", "age"])
        CsvSaver(dirname)(df.rdd)

        # read saved data and check it
        for root, d, files in os.walk(dirname):
            for f in files:
                filename = os.path.join(root, f)
                if filename.endswith(".csv"):
                    with open(filename) as f:
                        reader = csv.reader(f)
                        next(reader)
                        data = [r for r in reader]

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][0], rows[0][0])

    def test_parquet_saver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = tmpdir
        try:
            # load and save data
            rows = [("Alice", 1)]
            df = self.spark.createDataFrame(rows, ["name", "age"])
            ParquetSaver(dirname + "/", explain=True)(df.rdd)
            ParquetSaver(dirname + "2/")(df.rdd)

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
        row = FieldsSelector(fields=["name"], explain=True)(df.rdd).first()
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

    def test_language_selector(self):
        language_selector = LanguageSelector(languages=["XML", "YAML"], blacklist=True)
        df = language_selector(HeadFiles()(self.engine))
        langs = [x.lang for x in df.select("lang").distinct().collect()]
        self.assertEqual(langs, ["Markdown", "Gradle", "Text", "INI",
                                 "Batchfile", "Python", "Java", "Shell"])

        language_selector = LanguageSelector(languages=["Python", "Java"], blacklist=False)
        df = language_selector(HeadFiles()(self.engine))
        langs = [x.lang for x in df.select("lang").distinct().collect()]
        self.assertEqual(langs, ["Python", "Java"])


if __name__ == '__main__':
    unittest.main()
