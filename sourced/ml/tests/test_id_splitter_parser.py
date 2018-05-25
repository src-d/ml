import argparse
import unittest

from sourced.ml.cmd_entries.args import ArgumentDefaultsHelpFormatterNoNone
from sourced.ml.cmd_entries.id_splitter import add_id_splitter_arguments


class IdSplitterPipelineTest(unittest.TestCase):
    def test_parser(self):
        # normal launch
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_loc = "fake_input"
            output_loc = "fake_output"
            add_id_splitter_arguments(parser)
            arguments = "-i {} -o {}".format(identifiers_loc, output_loc)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_size"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "rate"))
            self.assertTrue(hasattr(args, "final_rate"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "num_chars"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_token"))
            self.assertTrue(hasattr(args, "csv_token_split"))
            self.assertTrue(hasattr(args, "csv_header"))

        except Exception as e:
            self.fail("parser raised {} with log {}".format(type(e), str(e)))

        # normal launch RNN
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_loc = "fake_input"
            output_loc = "fake_output"
            add_id_splitter_arguments(parser)
            arguments = "-i {} -o {} rnn".format(identifiers_loc, output_loc)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_size"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "rate"))
            self.assertTrue(hasattr(args, "final_rate"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "num_chars"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_token"))
            self.assertTrue(hasattr(args, "csv_token_split"))
            self.assertTrue(hasattr(args, "csv_header"))

            # RNN
            self.assertTrue(hasattr(args, "type"))
            self.assertTrue(hasattr(args, "neurons"))
            self.assertTrue(hasattr(args, "stack"))

        except Exception as e:
            self.fail("parser raised {} with log {}".format(type(e), str(e)))

        # normal launch CNN
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_loc = "fake_input"
            output_loc = "fake_output"
            add_id_splitter_arguments(parser)
            arguments = "-i {} -o {} cnn".format(identifiers_loc, output_loc)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_size"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "rate"))
            self.assertTrue(hasattr(args, "final_rate"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "num_chars"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_token"))
            self.assertTrue(hasattr(args, "csv_token_split"))
            self.assertTrue(hasattr(args, "csv_header"))

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
            identifiers_loc = "fake_input"
            add_id_splitter_arguments(parser)
            arguments = "-i {}".format(identifiers_loc)
            parser.parse_args(arguments.split())

        self.assertEqual(cm.exception.code, 2)

        # not normal launch - missed input
        with self.assertRaises(SystemExit) as cm:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            output_loc = "fake_output"
            add_id_splitter_arguments(parser)
            arguments = "-o {}".format(output_loc)
            parser.parse_args(arguments.split())

        self.assertEqual(cm.exception.code, 2)
