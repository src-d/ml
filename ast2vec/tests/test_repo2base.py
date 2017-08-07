import logging
import multiprocessing
import os
import tempfile
import unittest

from ast2vec.cloning import RepoCloner
import ast2vec.repo2.base
from ast2vec.repo2.base import Repo2Base, RepoTransformer, ensure_bblfsh_is_running_noexc


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

    def test_overwrite_existing(self):
        self.assertEqual(self.base.overwrite_existing, self.base.DEFAULT_OVERWRITE_EXISTING)
        self.base.overwrite_existing = False
        self.assertEqual(self.base.overwrite_existing, False)
        self.base.overwrite_existing = True
        self.assertEqual(self.base.overwrite_existing, True)
        with self.assertRaises(TypeError):
            self.base.threads = 1.5


class RepoTransformerTests(unittest.TestCase):
    def setUp(self):
        self.backup = RepoTransformer._get_log_name
        RepoTransformer._get_log_name = lambda _: "RepoTransformer"

        class WORKER:
            DEFAULT_OVERWRITE_EXISTING = False

        RepoTransformer.WORKER_CLASS = WORKER
        self.base = RepoTransformer()

    def tearDown(self):
        RepoTransformer._get_log_name = self.backup
        RepoTransformer.WORKER_CLASS = None

    def test_num_processes(self):
        self.assertEqual(self.base.num_processes, self.base.DEFAULT_NUM_PROCESSES)
        self.base.num_processes = 10
        self.assertEqual(self.base.num_processes, 10)
        with self.assertRaises(TypeError):
            self.base.num_processes = 1.5
        with self.assertRaises(ValueError):
            self.base.num_processes = -1

    def test_process_repo_already_exists(self):
        exists_ = os.path.exists
        info_ = self.base._log.info

        def exists(path):
            return path == "xxx"

        def info(msg, arg):
            self.assertEqual(msg, "Model %s already exists, skipping.")
            self.assertEqual(arg, "xxx")

        try:
            os.path.exists = exists
            self.base._log.info = info
            self.assertTrue(self.base.process_repo("yyy", "xxx"))
        finally:
            os.path.exists = exists_
            self.base._log.info = info_


class EnsureBblfshIsRunningNoexcTest(unittest.TestCase):
    def setUp(self):
        def ensure_bblfsh_is_running_raise(*args, **kwargs):
            raise Exception
        self.save = ast2vec.repo2.base.ensure_bblfsh_is_running
        ast2vec.repo2.base.ensure_bblfsh_is_running = ensure_bblfsh_is_running_raise

    def tearDown(self):
        ast2vec.repo2.base.ensure_bblfsh_is_running = self.save

    def test_ensure_bblfsh_is_running_raise(self):
        def log_check(msg):
            self.assertEqual(msg, "Failed to ensure that the Babelfish server is running.")

        log = logging.getLogger("bblfsh")
        exception_ = log.exception
        warning_ = log.warning
        try:
            log.setLevel(logging.WARNING)
            log.warning = log_check
            ensure_bblfsh_is_running_noexc()

            log.setLevel(logging.DEBUG)
            log.exception = log_check
            ensure_bblfsh_is_running_noexc()
        finally:
            logging.exception = exception_
            logging.warning = warning_


if __name__ == "__main__":
    unittest.main()
