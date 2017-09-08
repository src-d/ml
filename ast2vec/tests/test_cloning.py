import argparse
import copy
import json
import os
import subprocess
import tempfile
import unittest

from ast2vec.cloning import clone_repositories, RepoCloner
import ast2vec.tests as tests


class RepoClonerTests(unittest.TestCase):
    base_args = argparse.Namespace(
        log_level="INFO", input=["git://github.com/src-d/jgscm.git"],
        ignore=True, redownload=False, threads=1)
    repo_urls = [
        "https://whatever.site.com/cool/you.git",
        "git://github.com/cool/you.git",
        "www.bitbucket.org/cool/you\\\n",
        "http://sourceforge.net/cool/you\n",
        "code.google.com/cool/you",
        "http://jenkins-ci.org/cool/you.git\n"
    ]
    repo_names = [
        "cool&you_whatever.site.com",
        "cool&you_github.com",
        "cool&you_www.bitbucket.org",
        "cool&you_sourceforge.net",
        "cool&you_code.google.com",
        "cool&you_jenkins-ci.org"
    ]

    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_enry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = copy.copy(self.base_args)
            args.output = tmpdir
            args.linguist = "xxx"
            with self.assertRaises(FileNotFoundError):
                clone_repositories(args)

    def test_full_clone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = copy.copy(self.base_args)
            args.output = tmpdir
            args.linguist = None
            self.assertIsNone(clone_repositories(args))
            self._validate_clone(tmpdir, args)

    def test_classified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = copy.copy(self.base_args)
            args.output = tmpdir
            args.linguist = tests.ENRY
            self.assertIsNone(clone_repositories(args))
            self._validate_clone(tmpdir, args)

    def test_languages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = copy.copy(self.base_args)
            args.output = tmpdir
            args.linguist = tests.ENRY
            args.languages = ["Python"]
            self.assertIsNone(clone_repositories(args))
            self._validate_clone(tmpdir, args)

    def test_multiple_repositories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = copy.copy(self.base_args)
            args.output = tmpdir
            args.linguist = tests.ENRY
            args.languages = ["Python"]
            args.input.append(os.path.join(os.path.dirname(__file__), "test_repos_list.txt"))
            self.assertIsNone(clone_repositories(args))
            self._validate_clone(tmpdir, args)

    def test_prepare_repo_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = RepoCloner(linguist=None, redownload=True)
            for repo_url, repo_name in zip(self.repo_urls, self.repo_names):
                repo_url = rc._prepare_repo_url(repo_url)
                self.assertEqual(
                    os.path.join(tmpdir, repo_name),
                    rc._prepare_repo_dir(repo_url, tmpdir))

    def test_prepare_repo_url(self):
        results = [
            "https://whatever.site.com/cool/you.git",
            "git://github.com/cool/you.git",
            "https://www.bitbucket.org/cool/you",
            "http://sourceforge.net/cool/you",
            "https://code.google.com/cool/you",
            "http://jenkins-ci.org/cool/you.git",
        ]
        for correct_url, repo_url in zip(results, self.repo_urls):
            self.assertEqual(correct_url, RepoCloner._prepare_repo_url(repo_url))

    def test_repo_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = copy.copy(self.base_args)
            args.output = str.encode(tmpdir)
            args.linguist = None
            with self.assertRaises(TypeError):
                clone_repositories(args)

    def test_subprocess_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = RepoCloner(linguist=None, redownload=True)
            with self.assertRaises(subprocess.CalledProcessError):
                rc.clone_repo("https://not.a/git/repo", False, tmpdir)

    def _collect_repo_files(self, repo_dir):
        files = list()
        for dirname, dirnames, filenames in os.walk(str.encode(repo_dir)):
            for filename in filenames:
                full_filename = os.path.join(dirname, filename)
                files.append(full_filename)
        return files

    def _validate_clone(self, tmpdir, args):
        self.assertTrue(len(os.listdir(args.output)) > 0)

        if args.linguist:
            enry = args.linguist
            self.assertTrue(os.path.isfile(enry))
            self.assertEqual(os.stat(enry).st_mode & 0o555, 0o555)
            output = subprocess.check_output([enry, "-json", args.output])
            lang_files = json.loads(output.decode("utf-8"))

            files = [str.encode(os.path.join(args.output, f)) for files in
                     lang_files.values() for f in files]
            repo_files = self._collect_repo_files(args.output)
            self.assertEqual(set(files), set(repo_files))

            if hasattr(args, "languages"):
                self.assertEqual(lang_files.keys(), set(args.languages))

if __name__ == "__main__":
    unittest.main()
