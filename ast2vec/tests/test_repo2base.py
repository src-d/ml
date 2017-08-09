import multiprocessing
import tempfile
import unittest

from ast2vec.cloning import RepoCloner
from ast2vec.repo2.base import Repo2Base


class Repo2BaseTests(unittest.TestCase):
    def setUp(self):
        self.backup = Repo2Base._get_log_name, RepoCloner.find_linguist
        Repo2Base._get_log_name = lambda _: "repo2"
        RepoCloner.find_linguist = lambda _1, _2: None
        self.base = Repo2Base()

    def tearDown(self):
        Repo2Base._get_log_name, RepoCloner.find_linguist = self.backup

    def test_tempdir(self):
        self.base.tempdir = tempfile.gettempdir()
        self.assertEqual(self.base.tempdir, tempfile.gettempdir())
        with self.assertRaises(ValueError):
            self.base.tempdir = "/xxxyyyzzz123456"
        self.base.tempdir = None

    def test_bblfsh_endpoint(self):
        self.assertEqual(self.base.bblfsh_endpoint, "0.0.0.0:9432")

    def test_timeout(self):
        self.assertEqual(self.base.timeout, self.base.DEFAULT_BBLFSH_TIMEOUT)
        self.base.timeout = 100500.1
        self.assertEqual(self.base.timeout, 100500.1)
        with self.assertRaises(TypeError):
            self.base.timeout = "t"
        with self.assertRaises(ValueError):
            self.base.timeout = 0
        with self.assertRaises(ValueError):
            self.base.timeout = -1
        self.base.timeout = None

    def test_threads(self):
        self.assertEqual(self.base.threads, multiprocessing.cpu_count())
        self.base.threads = 11
        self.assertEqual(self.base.threads, 11)
        self.assertEqual(len(self.base._bblfsh), 11)
        with self.assertRaises(TypeError):
            self.base.threads = 1.5
        with self.assertRaises(ValueError):
            self.base.threads = 0


if __name__ == "__main__":
    unittest.main()
