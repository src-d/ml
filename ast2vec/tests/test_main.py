import argparse
import sys
import unittest


import ast2vec.__main__ as main

class MainTests(unittest.TestCase):
    def test_handlers(self):
        handlers = [False] * 11

        def repo2nbow_entry(args):
            handlers[0] = True

        def repos2nbow_entry(args):
            handlers[1] = True

        def repo2coocc_entry(args):
            handlers[2] = True

        def repos2coocc_entry(args):
            handlers[3] = True

        def preprocess(args):
            handlers[4] = True

        def run_swivel(args):
            handlers[5] = True

        def postprocess(args):
            handlers[6] = True

        def install_enry(args):
            handlers[7] = True

        def dump_model(args):
            handlers[8] = True

        def publish_model(args):
            handlers[9] = True

        def list_models(args):
            handlers[10] = True

        main.repo2nbow_entry = repo2nbow_entry
        main.repos2nbow_entry = repos2nbow_entry
        main.repo2coocc_entry = repo2coocc_entry
        main.repos2coocc_entry = repos2coocc_entry
        main.preprocess = preprocess
        main.run_swivel = run_swivel
        main.postprocess = postprocess
        main.install_enry = install_enry
        main.dump_model = dump_model
        main.publish_model = publish_model
        main.list_models = list_models
        args = sys.argv
        error = argparse.ArgumentParser.error
        argparse.ArgumentParser.error = lambda self, message: None

        for action in ("repo2nbow", "repos2nbow", "repo2coocc", "repos2coocc",
                       "preproc", "train", "postproc", "enry", "dump",
                       "publish", "list-models"):
            sys.argv = [main.__file__, action]
            main.main()

        sys.argv = args
        argparse.ArgumentParser.error = error
        self.assertEqual(sum(handlers), 11)

if __name__ == "__main__":
    unittest.main()