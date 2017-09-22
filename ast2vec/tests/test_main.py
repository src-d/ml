import argparse
import sys
import unittest

import ast2vec.__main__ as main
from ast2vec.__main__ import ArgumentDefaultsHelpFormatterNoNone
from ast2vec.tests.test_dump import captured_output


class MainTests(unittest.TestCase):
    def test_handlers(self):
        action2handler = {
            "clone": "clone_repositories",
            "repo2nbow": "repo2nbow_entry",
            "repos2nbow": "repos2nbow_entry",
            "repo2coocc": "repo2coocc_entry",
            "repos2coocc": "repos2coocc_entry",
            "join-bow": "joinbow_entry",
            "repo2uast": "repo2uast_entry",
            "repos2uast": "repos2uast_entry",
            "repo2source": "repo2source_entry",
            "repos2source": "repos2source_entry",
            "uast2prox": "prox_entry",
            "uast2df": "uast2df_entry",
            "uast2bow": "uast2bow_entry",
            "id2vec_preproc": "preprocess_id2vec",
            "id2vec_train": "run_swivel",
            "id2vec_postproc": "postprocess_id2vec",
            "id2vec_projector": "projector_entry",
            "bigartm2asdf": "bigartm2asdf_entry",
            "bow2vw": "bow2vw_entry",
            "enry": "install_enry",
            "bigartm": "install_bigartm",
            "dump": "dump_model",
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

                handler_save = getattr(main, handler)
                try:
                    setattr(main, handler, handler_append)
                    sys.argv = [main.__file__, action]
                    main.main()
                finally:
                    setattr(main, handler, handler_save)
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
