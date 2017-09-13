import argparse
import json
import os
import subprocess
import tempfile
import unittest

from ast2vec.enry import install_enry


class EnryTests(unittest.TestCase):
    gitdir = os.path.join(os.path.dirname(__file__), "..", "..")

    def test_install_enry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(output=tmpdir, tmpdir=None, force_build=True)
            self.assertIsNone(install_enry(args))
            self._valivate_enry(tmpdir)

    def test_install_enry_no_args_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(install_enry(
                target=os.path.join(tmpdir, "enry")))
            self._valivate_enry(tmpdir)

    def _valivate_enry(self, tmpdir):
        enry = os.path.join(tmpdir, "enry")
        self.assertTrue(os.path.isfile(enry))
        self.assertEqual(os.stat(enry).st_mode & 0o555, 0o555)
        output = subprocess.check_output([enry, "-json", self.gitdir])
        files = json.loads(output.decode("utf-8"))
        self.assertIn("Python", files)


if __name__ == "__main__":
    unittest.main()
