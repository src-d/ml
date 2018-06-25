import argparse
import unittest

from sourced.ml.cmd.args import ArgumentDefaultsHelpFormatterNoNone
from sourced.ml.cmd_entries.id_splitter import add_train_id_splitter_args


class IdSplitterPipelineTest(unittest.TestCase):
    def test_parser(self):
        # normal launch
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_path = "fake_input"
            output_path = "fake_output"
            add_train_id_splitter_args(parser)
            arguments = "-i {} -o {}".format(identifiers_path, output_path)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_ratio"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "lr"))
            self.assertTrue(hasattr(args, "final_lr"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_identifier"))
            self.assertTrue(hasattr(args, "csv_identifier_split"))
            self.assertTrue(hasattr(args, "include_csv_header"))

        except Exception as e:
            self.fail("parser raised {} with log {}".format(type(e), str(e)))

        # normal launch RNN
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_path = "fake_input"
            output_path = "fake_output"
            add_train_id_splitter_args(parser)
            arguments = "-i {} -o {} rnn".format(identifiers_path, output_path)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_ratio"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "lr"))
            self.assertTrue(hasattr(args, "final_lr"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_identifier"))
            self.assertTrue(hasattr(args, "csv_identifier_split"))
            self.assertTrue(hasattr(args, "include_csv_header"))

            # RNN
            self.assertTrue(hasattr(args, "type"))
            self.assertTrue(hasattr(args, "neurons"))
            self.assertTrue(hasattr(args, "stack"))

        except Exception as e:
            self.fail("parser raised {} with log {}".format(type(e), str(e)))

        # normal launch CNN
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_path = "fake_input"
            output_path = "fake_output"
            add_train_id_splitter_args(parser)
            arguments = "-i {} -o {} cnn".format(identifiers_path, output_path)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_ratio"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "lr"))
            self.assertTrue(hasattr(args, "final_lr"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_identifier"))
            self.assertTrue(hasattr(args, "csv_identifier_split"))
            self.assertTrue(hasattr(args, "include_csv_header"))

            # CNN
            self.assertTrue(hasattr(args, "filters"))
            self.assertTrue(hasattr(args, "kernel_sizes"))
            self.assertTrue(hasattr(args, "stack"))
            self.assertTrue(hasattr(args, "dim_reduction"))

        except Exception as e:
            self.fail("parser raised {} with log {}".format(type(e), str(e)))

        # not normal launch - missed output
        with self.assertRaises(SystemExit) as cm:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_path = "fake_input"
            add_train_id_splitter_args(parser)
            arguments = "-i {}".format(identifiers_path)
            parser.parse_args(arguments.split())

        self.assertEqual(cm.exception.code, 2)

        # not normal launch - missed input
        with self.assertRaises(SystemExit) as cm:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            output_path = "fake_output"
            add_train_id_splitter_args(parser)
            arguments = "-o {}".format(output_path)
            parser.parse_args(arguments.split())

        self.assertEqual(cm.exception.code, 2)
