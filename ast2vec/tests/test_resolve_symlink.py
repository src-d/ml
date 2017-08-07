import os
import unittest

from ast2vec import resolve_symlink


class ResolveSymlinkTests(unittest.TestCase):
    def test_resolve_symlink(self):
        exists_ = os.path.exists
        islink_ = os.path.islink
        readlink_ = os.readlink
        try:
            def exists(path):
                return path == "yyy"

            def islink(path):
                return path == "xxx" or path == "zzz"

            def readlink(path):
                return "yyy" if path != "xxx" else path

            os.path.exists = exists
            os.path.islink = islink
            os.readlink = readlink

            self.assertEqual(resolve_symlink.resolve_symlink("zzz"), "yyy")
            with self.assertRaises(resolve_symlink.DanglingSymlinkError) as _:
                resolve_symlink.resolve_symlink("xxx")
        finally:
            os.path.exists = exists_
            os.path.islink = islink_
            os.readlink = readlink_

if __name__ == "__main__":
    unittest.main()
