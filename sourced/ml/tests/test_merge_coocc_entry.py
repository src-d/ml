import logging
import os
import shutil
import tempfile
import unittest

import numpy as np

from sourced.ml.cmd.merge_coocc import merge_coocc, load_and_check, MAX_INT32
from sourced.ml.models import Cooccurrences
from sourced.ml.tests import models

COPIES_NUMBER = 3


def get_args(_input_dir, _no_spark):
    class args:
        input = [os.path.join(_input_dir, x) for x in os.listdir(_input_dir)]
        output = os.path.join(_input_dir, "res_coocc.asdf")
        docfreq = models.COOCC_DF
        pause = False
        filter = "**/*.asdf"
        log_level = "INFO"
        no_spark = _no_spark
    return args


class MergeCooccEntry(unittest.TestCase):
    def check_coocc(self, output, copies_number=COPIES_NUMBER):
        coocc = Cooccurrences().load(models.COOCC)
        res = Cooccurrences().load(output)
        self.assertEqual(len(res.tokens), len(coocc.tokens))
        permutation = [coocc.tokens.index(token) for token in res.tokens]
        self.assertTrue(np.all(res.matrix.todense() ==
                               copies_number *
                               coocc.matrix.todense()[permutation][:, permutation]))

    def copy_models(self, model_path, to_dir, n):
        coocc_filename = os.path.split(model_path)[1]
        for i in range(n):
            shutil.copyfile(model_path,
                            os.path.join(to_dir, "{}.".format(i).join(coocc_filename.split("."))))

    def test_with_spark(self):
        with tempfile.TemporaryDirectory(prefix="merge-coocc-entry-test") as input_dir:
            self.copy_models(models.COOCC, input_dir, COPIES_NUMBER)
            args = get_args(input_dir, False)
            merge_coocc(args)
            self.check_coocc(args.output)

    def test_without_spark(self):
        with tempfile.TemporaryDirectory(prefix="merge-coocc-entry-test") as input_dir:
            self.copy_models(models.COOCC, input_dir, COPIES_NUMBER)
            args = get_args(input_dir, True)
            merge_coocc(args)
            self.check_coocc(args.output)

    def test_load_and_check(self):
        with tempfile.TemporaryDirectory(prefix="merge-coocc-entry-test") as input_dir:
            self.copy_models(models.COOCC, input_dir, COPIES_NUMBER)
            args = get_args(input_dir, True)
            c_neg = Cooccurrences().load(args.input[0])
            c_neg.matrix.data[0] = -1
            c_neg.save(args.input[0])
            self.assertEqual(len(list(load_and_check(args.input, logging.getLogger("test")))), 2)

            c_neg = Cooccurrences().load(args.input[0])
            c_neg.matrix.data = np.uint32(c_neg.matrix.data)
            c_neg.matrix.data[0] = MAX_INT32 + 1
            c_neg.save(args.input[0])
            for path, coocc in load_and_check(args.input, logging.getLogger("test")):
                self.assertTrue(np.all(coocc.matrix.data <= MAX_INT32))
                break

    def test_overflow_with_spark(self):
        with tempfile.TemporaryDirectory(prefix="merge-coocc-entry-test") as input_dir:
            self.copy_models(models.COOCC, input_dir, COPIES_NUMBER)
            args = get_args(input_dir, False)
            c_neg = Cooccurrences().load(args.input[0])
            c_neg.matrix.data[0] = MAX_INT32 - c_neg.matrix.data[0]
            c_neg.save(args.input[0])
            merge_coocc(args)

            result = Cooccurrences().load(args.output)
            self.assertTrue(np.all(result.matrix.data <= MAX_INT32))
            self.assertTrue(np.all(result.matrix.data >= 0))

    def test_overflow_without_spark(self):
        with tempfile.TemporaryDirectory(prefix="merge-coocc-entry-test") as input_dir:
            self.copy_models(models.COOCC, input_dir, 10)
            args = get_args(input_dir, True)
            c_neg = Cooccurrences().load(args.input[0])
            c_neg.matrix.data[0] = MAX_INT32 - 5 * c_neg.matrix.data[0]
            c_neg.save(args.input[0])
            merge_coocc(args)

            result = Cooccurrences().load(args.output)
            self.assertTrue(np.all(result.matrix.data <= MAX_INT32))
            self.assertTrue(np.all(result.matrix.data >= 0))


if __name__ == '__main__':
    unittest.main()
