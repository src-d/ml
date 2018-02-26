import argparse
import gzip
import os
from site import getsitepackages
import tempfile
from typing import NamedTuple
import unittest

from pyspark import Row

from sourced.ml.extractors import BagsExtractor
from sourced.ml.tests import tfidf_data
from sourced.ml.transformers import Content2Ids
from sourced.ml.utils import create_spark


class Content2IdsTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark("test")
        Column = NamedTuple("Column", [("repo_id", str), ("file_id", str)])
        self.column_names = Column(repo_id="document", file_id="file")
        self.language_mapping = Content2Ids.build_mapping()

    def test_languages(self):
        for path in getsitepackages():
            path_to_languages = os.path.join(path, "sourced/ml/transformers/languages.yml")
            if os.path.exists(path_to_languages):
                break
        self.assertTrue(os.path.exists(path_to_languages))

    def test_call(self):
        content2ids = Content2Ids(self.language_mapping, self.column_names,
                                  split=False, idfreq=False)
        for data, result in zip(tfidf_data.datasets, tfidf_data.ids_result):
            rdd = self.sc.sparkContext \
                .parallelize(range(len(data["content"]))) \
                .map(lambda x: Row(content=data["content"][x], file=str(data["file"][x]),
                                   document=str(data["document"][x]), lang=data["lang"][x]))
            self.assertEqual(result, set(content2ids(rdd).collect()))

    def test_call_split(self):
        content2ids = Content2Ids(self.language_mapping, self.column_names,
                                  split=True, idfreq=False)
        for data, result in zip(tfidf_data.datasets, tfidf_data.ids_split_result):
            rdd = self.sc.sparkContext \
                .parallelize(range(len(data["content"]))) \
                .map(lambda x: Row(content=data["content"][x], file=str(data["file"][x]),
                                   document=str(data["document"][x]), lang=data["lang"][x]))
            self.assertEqual(result, set(content2ids(rdd).collect()))

    def test_call_split_idfreq(self):
        content2ids = Content2Ids(self.language_mapping, self.column_names,
                                  split=True, idfreq=True)
        for data, result in zip(tfidf_data.datasets, tfidf_data.ids_split_idfreq_result):
            rdd = self.sc.sparkContext \
                .parallelize(range(len(data["content"]))) \
                .map(lambda x: Row(content=data["content"][x], file=str(data["file"][x]),
                                   document=str(data["document"][x]), lang=data["lang"][x]))
            self.assertEqual(result, set(content2ids(rdd).collect()))

    def test_process_row_split(self):
        content2ids = Content2Ids(self.language_mapping, self.column_names,
                                  split=True, idfreq=False)
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
        content2ids = Content2Ids(self.language_mapping, self.column_names,
                                  split=False, idfreq=False)
        row = Row(content="from foo import FooBar; print('foobar')",
                  file="foobar.py",
                  document="src-d/ml",
                  lang="Python")
        self.assertEqual(list(content2ids._process_row(row)),
                         [("foo", ("src-d/ml", "src-d/ml/foobar.py")),
                          ("FooBar", ("src-d/ml", "src-d/ml/foobar.py"))])


if __name__ == "__main__":
    unittest.main()
