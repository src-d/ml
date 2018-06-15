import argparse
import sys
import unittest

import sourced.ml.__main__ as main
from sourced.ml.cmd.args import ArgumentDefaultsHelpFormatterNoNone

from sourced.ml.tests.test_dump import captured_output


class MainTests(unittest.TestCase):
    def test_handlers(self):
        action2handler = {
            "id2vec-preproc": "id2vec_preprocess",
            "id2vec-train": "run_swivel",
            "id2vec-postproc": "id2vec_postprocess",
            "id2vec-project": "id2vec_project",
            "bigartm2asdf": "bigartm2asdf",
            "bow2vw": "bow2vw",
            "bigartm": "install_bigartm",
            "dump": "dump_model",
            "repos2coocc": "repos2coocc",
            "repos2df": "repos2df",
            "repos2ids": "repos2ids",
            "repos2bow": "repos2bow",
            "repos2roleids": "repos2roles_and_ids",
            "repos2id_distance": "repos2id_distance",
            "repos2idseq": "repos2id_sequence",
            "preprocrepos": "preprocess_repos",
            "merge-df": "merge_df",
            "merge-coocc": "merge_coocc",
        }
        parser = main.get_parser()
        subcommands = set([x.dest for x in parser._subparsers._actions[2]._choices_actions])
        set_action2handler = set(action2handler)
        self.assertFalse(len(subcommands - set_action2handler),
                         "You forgot to add to this test {} subcommand(s) check".format(
                             subcommands - set_action2handler))

        self.assertFalse(len(set_action2handler - subcommands),
                         "You cover unexpected subcommand(s) {}".format(
                             set_action2handler - subcommands))

        called_actions = []
        args_save = sys.argv
        error_save = argparse.ArgumentParser.error
        try:
            argparse.ArgumentParser.error = lambda self, message: None

            for action, handler in action2handler.items():
                def handler_append(*args, **kwargs):
                    called_actions.append(action)

                module = main.cmd if hasattr(main.cmd, handler) else main
                handler_save = getattr(module, handler)
                try:
                    setattr(module, handler, handler_append)
                    sys.argv = [main.__file__, action]
                    main.main()
                finally:
                    setattr(module, handler, handler_save)
        finally:
            sys.argv = args_save
            argparse.ArgumentParser.error = error_save

        set_called_actions = set(called_actions)
        set_actions = set(action2handler)
        self.assertEqual(set_called_actions, set_actions)
        self.assertEqual(len(set_called_actions), len(called_actions))

    def test_empty(self):
        args = sys.argv
        error = argparse.ArgumentParser.error
        try:
            argparse.ArgumentParser.error = lambda self, message: None

            sys.argv = [main.__file__]
            with captured_output() as (stdout, _, _):
                main.main()
        finally:
            sys.argv = args
            argparse.ArgumentParser.error = error
        self.assertIn("usage:", stdout.getvalue())

    def test_custom_formatter(self):
        class FakeAction:
            default = None
            option_strings = ['--param']
            nargs = None
            help = "help"

        formatter = ArgumentDefaultsHelpFormatterNoNone(None)
        help = formatter._expand_help(FakeAction)
        self.assertEqual("help", help)
        FakeAction.default = 10
        help = formatter._expand_help(FakeAction)
        self.assertEqual("help (default: 10)", help)


if __name__ == "__main__":
    unittest.main()
