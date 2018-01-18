import argparse
import logging
import os
import sys
import unittest
from contextlib import contextmanager
from io import StringIO

import modelforge.gcs_backend as gcs_backend
import sourced.ml.tests.models as paths
from sourced.ml.tests.fake_requests import FakeRequests
from sourced.ml.cmd_entries import dump_model


@contextmanager
def captured_output():
    log = StringIO()
    log_handler = logging.StreamHandler(log)
    logging.getLogger().addHandler(log_handler)
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr, log
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        logging.getLogger().removeHandler(log_handler)


class DumpTests(unittest.TestCase):
    ID2VEC_DUMP = """{'created_at': datetime.datetime(2017, 6, 18, 17, 37, 6, 255615),
 'dependencies': [],
 'model': 'id2vec',
 'uuid': '92609e70-f79c-46b5-8419-55726e873cfc',
 'version': [1, 0, 0]}
Shape: (1000, 300)
First 10 words: ['get', 'name', 'type', 'string', 'class', 'set', 'data', 'value', 'self', 'test']
"""
    DOCFREQ_DUMP = """{'created_at': datetime.datetime(2017, 8, 9, 16, 49, 12, 775367),
 'dependencies': [],
 'model': 'docfreq',
 'uuid': 'f64bacd4-67fb-4c64-8382-399a8e7db52a',
 'version': [0, 1, 0]}
Number of words: 982
""" + "First 10 words: ['aaa', 'aaaa', 'aaaaa', 'aaaaaa', 'aaaaaaa', 'aaaaaaaa', 'aaaaaaaaa', " \
      "'aaaaaaaaaa', 'aaaaaaaaaaa', 'aaaaaaaaaaaa']\nNumber of documents: 1000\n"

    BOW_DUMP = """{'created_at': datetime.datetime(2018, 1, 18, 21, 59, 59, 200818),
 'dependencies': [{'created_at': datetime.datetime(2018, 1, 18, 21, 59, 48, 828287),
                   'dependencies': [],
                   'model': 'ordered_docfreq',
                   'uuid': '2c4fcae7-93a6-496e-9e3a-d6e15d35b812',
                   'version': [1, 0, 0]}],
 'model': 'bow',
 'parent': 'a65afc98-80d4-4b04-8353-9037e61adf2b',
 'uuid': 'd221c3c6-55bb-46f9-a807-cc3351e46bbb',
 'version': [1, 0, 10]}
Shape: (5, 20)
First 10 documents: ['repo1', 'repo2', 'repo3', 'repo4', 'repo5']
First 10 tokens: ['i.', 'i.*', 'i.Activity', 'i.AdapterView', 'i.ArrayAdapter', 'i.Arrays', 'i.Bundle', 'i.EditText', 'i.Exception', 'i.False']\n"""  # nopep8

    COOCC_DUMP = """{'created_at': datetime.datetime(2017, 7, 5, 18, 4, 5, 688259),
 'dependencies': [],
 'model': 'co-occurrences',
 'uuid': '215aadce-d98c-4391-b93f-90cae582e895',
 'version': [1, 0, 0]}
Number of words: 394
""" + ("First 10 words: ['generic', 'model', 'dump', 'printer', 'pprint', 'print', 'nbow', 'vec', 'idvec', 'coocc']\n" +  # nopep8
"""Matrix: shape: (394, 394) non-zero: 20832
""")

    def test_id2vec(self):
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input=self._get_path(paths.ID2VEC)))
        self.assertEqual(out.getvalue(), self.ID2VEC_DUMP)

    def test_docfreq(self):
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input=self._get_path(paths.DOCFREQ)))
        self.assertEqual(out.getvalue(), self.DOCFREQ_DUMP)

    def test_bow(self):
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input=self._get_path(paths.BOW)))
        self.assertEqual(out.getvalue(), self.BOW_DUMP)

    def test_coocc(self):
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input=self._get_path(paths.COOCC)))
        self.assertEqual(out.getvalue(), self.COOCC_DUMP)

    def test_id2vec_id(self):
        def route(url):
            if gcs_backend.INDEX_FILE in url:
                return '{"models": {"id2vec": {' \
                       '"92609e70-f79c-46b5-8419-55726e873cfc": ' \
                       '{"url": "https://xxx"}}}}'.encode()
            self.assertEqual("https://xxx", url)
            with open(self._get_path(paths.ID2VEC), "rb") as fin:
                return fin.read()

        gcs_backend.requests = FakeRequests(route)
        with captured_output() as (out, err, _):
            dump_model(self._get_args(
                input="92609e70-f79c-46b5-8419-55726e873cfc"))
        self.assertEqual(out.getvalue(), self.ID2VEC_DUMP)
        self.assertFalse(err.getvalue())

    def test_id2vec_url(self):
        def route(url):
            self.assertEqual("https://xxx", url)
            with open(self._get_path(paths.ID2VEC), "rb") as fin:
                return fin.read()

        gcs_backend.requests = FakeRequests(route)
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input="https://xxx"))
        self.assertEqual(out.getvalue(), self.ID2VEC_DUMP)

    def test_gcs(self):
        def route(url):
            if gcs_backend.INDEX_FILE in url:
                self.assertIn("custom", url)
                return '{"models": {"id2vec": {' \
                       '"92609e70-f79c-46b5-8419-55726e873cfc": ' \
                       '{"url": "https://xxx"}}}}'.encode()
            self.assertEqual("https://xxx", url)
            with open(self._get_path(paths.ID2VEC), "rb") as fin:
                return fin.read()

        gcs_backend.requests = FakeRequests(route)
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input="92609e70-f79c-46b5-8419-55726e873cfc",
                                      gcs="custom"))
        self.assertEqual(out.getvalue(), self.ID2VEC_DUMP)

    @staticmethod
    def _get_args(input=None, gcs=None, dependency=tuple()):
        return argparse.Namespace(input=input, gcs_bucket=gcs, dependency=dependency,
                                  log_level="WARNING")

    @staticmethod
    def _get_path(name):
        return os.path.join(os.path.dirname(__file__), name)


if __name__ == "__main__":
    unittest.main()
