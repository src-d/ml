import argparse
import gzip
import tempfile
import unittest

from pyspark import Row

from sourced.ml.extractors import BagsExtractor
from sourced.ml.tests import tfidf_data
from sourced.ml.transformers import Content2Ids
from sourced.ml.utils import create_spark


class Content2IdsTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark("test")
        self.documents_column = ["document", "file"]

    def test_call(self):
        args = argparse.Namespace(split=False, idfreq=False, output=None)
        content2ids = Content2Ids(args, self.documents_column)
        for data, result in zip(tfidf_data.datasets, tfidf_data.ids_result):
            rdd = self.sc.sparkContext \
                .parallelize(range(len(data["content"]))) \
                .map(lambda x: Row(content=data["content"][x], file=str(data["file"][x]),
                                   document=str(data["document"][x]), lang=data["lang"][x]))
            self.assertEqual(result, set(content2ids(rdd).collect()))

    def test_call_split(self):
        args = argparse.Namespace(split=True, idfreq=False, output=None)
        content2ids = Content2Ids(args, self.documents_column)
        for data, result in zip(tfidf_data.datasets, tfidf_data.ids_split_result):
            rdd = self.sc.sparkContext \
                .parallelize(range(len(data["content"]))) \
                .map(lambda x: Row(content=data["content"][x], file=str(data["file"][x]),
                                   document=str(data["document"][x]), lang=data["lang"][x]))
            self.assertEqual(result, set(content2ids(rdd).collect()))

    def test_call_split_idfreq(self):
        args = argparse.Namespace(split=True, idfreq=True, output=None)
        content2ids = Content2Ids(args, self.documents_column)
        for data, result in zip(tfidf_data.datasets, tfidf_data.ids_split_idfreq_result):
            rdd = self.sc.sparkContext \
                .parallelize(range(len(data["content"]))) \
                .map(lambda x: Row(content=data["content"][x], file=str(data["file"][x]),
                                   document=str(data["document"][x]), lang=data["lang"][x]))
            self.assertEqual(result, set(content2ids(rdd).collect()))

    def test_process_row_split(self):
        args = argparse.Namespace(split=True, idfreq=False, output=None)
        content2ids = Content2Ids(args, self.documents_column)
        content2ids.build_mapping()
        row = Row(content="from foo import FooBar; print('foobar')",
                  file="foobar.py",
                  document="src-d/ml",
                  lang="Python")
        row_md = Row(content="#title",
                     file="README",
                     document="src-d/ml",
                     lang="Markdown")
        row_batch = Row(content="@ECHO OFF; ECHO Hello World!",
                        file="bat",
                        document="src-d/ml",
                        lang="Batchfile")
        self.assertEqual(list(content2ids._process_row(row)),
                         [("FooBar", ("src-d/ml", "src-d/ml/foobar.py"))])
        self.assertEqual(len(list(content2ids._process_row(row_md))), 0)
        self.assertEqual(len(list(content2ids._process_row(row_batch))), 0)

    def test_process_row(self):
        args = argparse.Namespace(split=False, idfreq=False, output=None)
        content2ids = Content2Ids(args, self.documents_column)
        content2ids.build_mapping()
        row = Row(content="from foo import FooBar; print('foobar')",
                  file="foobar.py",
                  document="src-d/ml",
                  lang="Python")
        self.assertEqual(list(content2ids._process_row(row)),
                         [("foo", ("src-d/ml", "src-d/ml/foobar.py")),
                          ("FooBar", ("src-d/ml", "src-d/ml/foobar.py"))])

    def test_save(self):
        with tempfile.NamedTemporaryFile(prefix="repos2ids-test-save", suffix=".csv.gz") as tmpf:
            args = argparse.Namespace(split=True, idfreq=True, output=tmpf.name)
            content2ids = Content2Ids(args, self.documents_column)
            rdd = self.sc.sparkContext \
                .parallelize(range(1)) \
                .map(lambda x: Row(token="FooBar", token_split="foo bar",
                                   num_repos=1, num_files=2, num_occ=3))
            content2ids.save(rdd)
            with gzip.open(tmpf, "r") as g:
                lines = g.readlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0].decode().split(",")), 5)


if __name__ == "__main__":
    unittest.main()
