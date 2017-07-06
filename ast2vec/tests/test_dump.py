import argparse
from contextlib import contextmanager
from io import StringIO
import os
import sys
import unittest

import ast2vec.model as model
from ast2vec.dump import dump_model
import ast2vec.tests.models as paths
from ast2vec.tests.fake_requests import FakeRequests


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class DumpTests(unittest.TestCase):
    def test_id2vec(self):
        with captured_output() as (out, err):
            dump_model(self._get_args(input=self._get_path(paths.ID2VEC)))
        reference = """{'created_at': datetime.datetime(2017, 6, 18, 17, 37, 6, 255615),
 'dependencies': [],
 'model': 'id2vec',
 'uuid': '92609e70-f79c-46b5-8419-55726e873cfc',
 'version': [1, 0, 0]}
Shape: (1000, 300)
First 10 words: ['get', 'name', 'type', 'string', 'class', 'set', 'data', 'value', 'self', 'test']
"""
        self.assertEqual(out.getvalue(), reference)

    def test_docfreq(self):
        with captured_output() as (out, err):
            dump_model(self._get_args(input=self._get_path(paths.DOCFREQ)))
        reference = """{'created_at': datetime.datetime(2017, 6, 19, 9, 59, 14, 766638),
 'dependencies': [],
 'model': 'docfreq',
 'uuid': 'f64bacd4-67fb-4c64-8382-399a8e7db52a',
 'version': [1, 0, 0]}
Number of words: 1000
First 10 words: ['aaa', 'aaaa', 'aaaaa', 'aaaaaa', 'aaaaaaa', 'aaaaaaaa', 'aaaaaaaaa', 'aaaaaaaaaa', 'aaaaaaaaaaa', 'aaaaaaaaaaaa']
"""
        self.assertEqual(out.getvalue(), reference)

    def test_nbow(self):
        with captured_output() as (out, err):
            dump_model(self._get_args(input=self._get_path(paths.NBOW)))
        reference = """{'created_at': datetime.datetime(2017, 6, 19, 9, 16, 8, 942880),
 'dependencies': [{'created_at': datetime.datetime(2017, 6, 18, 17, 37, 6, 255615),
                   'dependencies': [],
                   'model': 'id2vec',
                   'uuid': '92609e70-f79c-46b5-8419-55726e873cfc',
                   'version': [1, 0, 0]},
                  {'created_at': datetime.datetime(2017, 6, 19, 9, 59, 14, 766638),
                   'dependencies': [],
                   'model': 'docfreq',
                   'uuid': 'f64bacd4-67fb-4c64-8382-399a8e7db52a',
                   'version': [1, 0, 0]}],
 'model': 'nbow',
 'uuid': '1e3da42a-28b6-4b33-94a2-a5671f4102f4',
 'version': [1, 0, 0]}
Shape: [1000, 999424]
First 10 repos: ['ikizir/HohhaDynamicXOR', 'ditesh/node-poplib', 'Code52/MarkPadRT', 'wp-shortcake/shortcake', 'capaj/Moonridge', 'HugoGiraudel/hugogiraudel.github.com', 'crosswalk-project/crosswalk-website', 'apache/parquet-mr', 'dciccale/kimbo.js', 'processone/oneteam']
"""
        self.assertEqual(out.getvalue(), reference)

    def test_coocc(self):
        with captured_output() as (out, err):
            dump_model(self._get_args(input=self._get_path(paths.COOCC)))
        reference = """{'created_at': datetime.datetime(2017, 7, 5, 18, 4, 5, 688259),
 'dependencies': [],
 'model': 'co-occurrences',
 'uuid': '215aadce-d98c-4391-b93f-90cae582e895',
 'version': [1, 0, 0]}
Number of words: 394
First 10 words: ['generic', 'model', 'dump', 'printer', 'pprint', 'print', 'nbow', 'vec', 'idvec', 'coocc']
Matrix: , shape: [394, 394] number of non zero elements 20832
"""
        self.assertEqual(out.getvalue(), reference)

    def test_id2vec_id(self):
        def route(url):
            if url.endswith(model.Model.INDEX_FILE):
                return '{"models": {"id2vec": {' \
                       '"92609e70-f79c-46b5-8419-55726e873cfc": ' \
                       '{"url": "https://xxx"}}}}'.encode()
            self.assertEqual("https://xxx", url)
            with open(self._get_path(paths.ID2VEC), "rb") as fin:
                return fin.read()

        model.requests = FakeRequests(route)
        with captured_output() as (out, err):
            dump_model(self._get_args(
                input="92609e70-f79c-46b5-8419-55726e873cfc"))
        reference = """{'created_at': datetime.datetime(2017, 6, 18, 17, 37, 6, 255615),
 'dependencies': [],
 'model': 'id2vec',
 'uuid': '92609e70-f79c-46b5-8419-55726e873cfc',
 'version': [1, 0, 0]}
Shape: (1000, 300)
First 10 words: ['get', 'name', 'type', 'string', 'class', 'set', 'data', 'value', 'self', 'test']
"""
        self.assertEqual(out.getvalue(), reference)

    @staticmethod
    def _get_args(input=None, gcs=None, dependency=tuple()):
        return argparse.Namespace(input=input, gcs=gcs, dependency=dependency)

    @staticmethod
    def _get_path(name):
        return os.path.join(os.path.dirname(__file__), name)


if __name__ == "__main__":
    unittest.main()
