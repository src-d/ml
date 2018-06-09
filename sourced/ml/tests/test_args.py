from io import StringIO
import sys
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

        self.assertEqual(set(files),
                         set(handle_input_arg(files)))
        self.assertEqual({files[0]},
                         set(handle_input_arg(files[0])))


if __name__ == "__main__":
    unittest.main()
