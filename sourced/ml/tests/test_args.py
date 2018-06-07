from io import StringIO
import os
import sys
import tempfile
import unittest


from sourced.ml.cmd.args import handle_input_arg


class DocumentFrequenciesTests(unittest.TestCase):

    def test_handle_input_arg(self):
        files = ["file1.asdf", "file2.asdf", "file3.asdf"]
        files_stdin = StringIO("\n".join(files))
        _stdin = sys.stdin
        try:
            sys.stdin = files_stdin
            self.assertEqual(files, list(handle_input_arg("-")))
        finally:
            sys.stdin = _stdin

        with tempfile.TemporaryDirectory(prefix="srcdml_handle_input_arg_") as dirname:
            for filename in files:
                open(os.path.join(dirname, filename), 'a').close()
            self.assertEqual(set(files),
                             set(os.path.split(x)[1] for x in
                                 handle_input_arg(dirname)))
            self.assertEqual({"file1.asdf"},
                             set(os.path.split(x)[1] for x in
                                 handle_input_arg(dirname, filter_arg="*1*")))


if __name__ == "__main__":
    unittest.main()
