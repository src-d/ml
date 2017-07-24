import os
import tempfile
import unittest
from subprocess import CalledProcessError

import ast2vec.tests as tests
from ast2vec.repo2.source import Repo2Source

repo_url = "https://github.com/src-d/ast2vec"  # Suppose that we will not change it :)


class Repo2BaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_prepare_reponame(self):
        reponames = [
            "https://whatever.cite.com/cool/you.git",
            "whatever.cite.com/cool/you.git",
            "whatever.cite.com/cool/you\\\n",
            "whatever.cite.com/cool/you\n",
            "whatever.cite.com/cool/you",
            "whatever.cite.com/cool/you.git\n",
        ]
        results = [
            "https://whatever.cite.com/cool/you.git",
            "https://whatever.cite.com/cool/you.git",
            "https://whatever.cite.com/cool/you",
            "https://whatever.cite.com/cool/you",
            "https://whatever.cite.com/cool/you",
            "https://whatever.cite.com/cool/you.git",
        ]
        for correct, repo in zip(results, reponames):
            self.assertEqual(correct, Repo2Source.prepare_reponame(repo))

    def test_clone_repository(self):
        r2b = Repo2Source(linguist=tests.ENRY)
        r2b.clone_repository(repo_url)
        # ok if no exception
        with tempfile.TemporaryDirectory() as tmp:
            r2b.clone_repository(repo_url, tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'README.md')))
            # ok if no exception
        with self.assertRaises(CalledProcessError) as context:
            # some bad one
            r2b.clone_repository(repo_url + "6bIUt676B6")


if __name__ == "__main__":
    unittest.main()
