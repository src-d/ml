import io
import tarfile
import tempfile
import unittest

import numpy

from sourced.ml.algorithms.id_splitter.features import prepare_features, read_identifiers
from sourced.ml.tests.models import IDENTIFIERS


def write_fake_identifiers(tar_file, n_lines, char_sizes, n_cols, text="a"):
    """
    Prepare file with fake identifiers.
    :param tar_file: ready to write file.
    :param n_lines: number of lines to genrate.
    :param char_sizes: sizes of identifiers.
    :param n_cols: number of columns.
    :param text: text that is used to fill identifiers.
    """
    # sanity check
    if isinstance(char_sizes, int):
        char_sizes = [char_sizes] * n_lines
    assert len(char_sizes) == n_lines

    # generate file
    res = []
    for sz in char_sizes:
        line = ",".join([text * sz] * n_cols)
        res.append(line)
    content = "\n".join(res)
    content = content.encode("utf-8")

    # add content to file
    info = tarfile.TarInfo("identifiers.txt")
    info.size = len(content)
    tar_file.addfile(info, io.BytesIO(content))


class IdSplitterTest(unittest.TestCase):
    def test_prepare_features(self):
        # check feature extraction
        text = "a a"
        n_lines = 10
        max_identifier_len = 20
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=n_lines, char_sizes=1, n_cols=2, text=text)
            feat = prepare_features(csv_path=tmp.name, use_header=True, identifiers_col=0,
                                    max_identifier_len=max_identifier_len, split_identifiers_col=1,
                                    shuffle=True, test_size=0.5, padding="post")
            x_train, x_test, y_train, y_test = feat
            # because of test_size=0.5 - shapes should be equal
            self.assertEqual(x_test.shape, x_train.shape)
            self.assertEqual(y_test.shape, y_train.shape)
            # each line contains only one split -> so it should be only 5 nonzero for train/test
            self.assertEqual(numpy.sum(y_test), 5)
            self.assertEqual(numpy.sum(y_train), 5)
            # each line contains only two chars -> so it should be only 10 nonzero for train/test
            self.assertEqual(numpy.count_nonzero(x_test), 10)
            self.assertEqual(numpy.count_nonzero(x_train), 10)
            # y should be 3 dimensional matrix
            self.assertEqual(y_test.ndim, 3)
            self.assertEqual(y_train.ndim, 3)
            # x should be 2 dimensional matrix
            self.assertEqual(x_test.ndim, 2)
            self.assertEqual(x_train.ndim, 2)
            # check number of samples
            self.assertEqual(x_test.shape[0] + x_train.shape[0], n_lines)
            self.assertEqual(y_test.shape[0] + y_train.shape[0], n_lines)
            # check max_identifier_len
            self.assertEqual(x_test.shape[1], max_identifier_len)
            self.assertEqual(x_train.shape[1], max_identifier_len)
            self.assertEqual(y_test.shape[1], max_identifier_len)
            self.assertEqual(y_train.shape[1], max_identifier_len)

        # normal file
        try:
            prepare_features(csv_path=IDENTIFIERS)
        except Exception as e:
            self.fail("prepare_features raised %s with log %s",  type(e), str(e))

    def test_read_identifiers(self):
        # read with header
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=1, n_cols=5)

            res = read_identifiers(csv_path=tmp.name, use_header=True)
            self.assertEqual(len(res), 10)

        # read without header
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=1, n_cols=5)

            res = read_identifiers(csv_path=tmp.name, use_header=False)
            self.assertEqual(len(res), 9)

        # read with max_identifier_len equal to 0 -> expect empty list
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=1, n_cols=5)

            res = read_identifiers(csv_path=tmp.name, max_identifier_len=0)
            self.assertEqual(len(res), 0)

        # generate temporary file with identifiers of specific lengths and filter by length
        char_sizes = list(range(1, 11))

        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=char_sizes, n_cols=5)

            # check filtering
            # read last two columns as identifiers
            for i in range(11):
                res = read_identifiers(csv_path=tmp.name, max_identifier_len=i, identifiers_col=3,
                                       split_identifiers_col=4)
                self.assertEqual(len(res), i)

        # read wrong columns
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=char_sizes, n_cols=2)

            with self.assertRaises(IndexError) as cm:
                read_identifiers(csv_path=tmp.name, max_identifier_len=10, identifiers_col=3,
                                 split_identifiers_col=4)

        # normal file
        try:
            read_identifiers(csv_path=IDENTIFIERS)
        except Exception as e:
            self.fail("read_identifiers raised %s with log %s", type(e), str(e))
