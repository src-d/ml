import argparse
import os
import tempfile
import unittest

from sourced.ml.cmd import train_id_split


class TrainIdSplitTest(unittest.TestCase):
    def setUp(self):
        self.input = os.path.join(os.path.dirname(__file__), "identifiers.csv.tar.gz")

    def test_id_split_train(self):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                args = argparse.Namespace(input=self.input, output=tmpdir, model="CNN",
                                          devices="-1", test_ratio=0.2, padding="post",
                                          optimizer="Adam", batch_size=2, val_batch_size=2,
                                          length=10, dim_reduction=2, epochs=1,
                                          samples_before_report=10, lr=0.001,
                                          final_lr=0.00001, seed=1989, csv_identifier=3,
                                          csv_identifier_split=4, stack=2, include_csv_header=True,
                                          filters="64,32,16,8", kernel_sizes="2,4,8,16")
                train_id_split(args)
        except Exception as e:
            self.fail("CNN training raised %s with log: %s" % (type(e), str(e)))


if __name__ == "__main__":
    unittest.main()
