import os
import shutil
import tempfile
import unittest

import numpy as np

from sourced.ml.cmd import merge_coocc_entry
from sourced.ml.models import Cooccurrences
from sourced.ml.tests import models


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
    def check_coocc(self, output):
        coocc = Cooccurrences().load(models.COOCC)
        res = Cooccurrences().load(output)
        self.assertEqual(len(res.tokens), len(coocc.tokens))
        permutation = [coocc.tokens.index(token) for token in res.tokens]
        self.assertTrue(np.all(res.matrix.todense() ==
                               3 * coocc.matrix.todense()[permutation][:, permutation]))

    def test_with_spark(self):
        with tempfile.TemporaryDirectory(prefix="merge-coocc-entry-test-") as input_dir:
            coocc_filename = os.path.split(models.COOCC)[1]
            shutil.copyfile(models.COOCC,
                            os.path.join(input_dir, coocc_filename))
            shutil.copyfile(models.COOCC,
                            os.path.join(input_dir, "2.".join(coocc_filename.split("."))))
            shutil.copyfile(models.COOCC,
                            os.path.join(input_dir, "3.".join(coocc_filename.split("."))))

            args = get_args(input_dir, False)
            merge_coocc_entry(args)
            self.check_coocc(args.output)

    def test_without_spark(self):
        with tempfile.TemporaryDirectory(prefix="merge-coocc-entry-test-") as input_dir:
            coocc_filename = os.path.split(models.COOCC)[1]
            shutil.copyfile(models.COOCC,
                            os.path.join(input_dir, coocc_filename))
            shutil.copyfile(models.COOCC,
                            os.path.join(input_dir, "2.".join(coocc_filename.split("."))))
            shutil.copyfile(models.COOCC,
                            os.path.join(input_dir, "3.".join(coocc_filename.split("."))))

            args = get_args(input_dir, True)
            merge_coocc_entry(args)
            self.check_coocc(args.output)


if __name__ == '__main__':
    unittest.main()
