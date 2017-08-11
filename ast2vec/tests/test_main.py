import argparse
import sys
import unittest


import ast2vec.__main__ as main
from ast2vec.tests.test_dump import captured_output


class MainTests(unittest.TestCase):
    def test_handlers(self):
        handlers = [False] * 17

        def repo2nbow_entry(args):
            handlers[0] = True

        def repos2nbow_entry(args):
            handlers[1] = True

        def repo2coocc_entry(args):
            handlers[2] = True

        def repos2coocc_entry(args):
            handlers[3] = True

        def repo2uast_entry(args):
            handlers[4] = True

        def repos2uast_entry(args):
            handlers[5] = True

        def joinbow_entry(args):
            handlers[6] = True

        def prox_entry(args):
            handlers[7] = True

        def source2df_entry(args):
            handlers[8] = True

        def source2bow_entry(args):
            handlers[9] = True

        def preprocess(args):
            handlers[10] = True

        def run_swivel(args):
            handlers[11] = True

        def postprocess(args):
            handlers[12] = True

        def bow2vw_entry(args):
            handlers[13] = True

        def install_enry(args):
            handlers[14] = True

        def install_bigartm(args):
            handlers[15] = True

        def dump_model(args):
            handlers[16] = True

        main.repo2nbow_entry = repo2nbow_entry
        main.repos2nbow_entry = repos2nbow_entry
        main.repo2coocc_entry = repo2coocc_entry
        main.repos2coocc_entry = repos2coocc_entry
        main.joinbow_entry = joinbow_entry
        main.source2df_entry = source2df_entry
        main.source2bow_entry = source2bow_entry
        main.preprocess = preprocess
        main.run_swivel = run_swivel
        main.postprocess = postprocess
        main.bow2vw_entry = bow2vw_entry
        main.install_enry = install_enry
        main.install_bigartm = install_bigartm
        main.dump_model = dump_model
        main.prox_entry = prox_entry
        main.repo2uast_entry = repo2uast_entry
        main.repos2uast_entry = repos2uast_entry
        args = sys.argv
        error = argparse.ArgumentParser.error
        argparse.ArgumentParser.error = lambda self, message: None

        for action in ("repo2nbow", "repos2nbow", "repo2coocc", "repos2coocc", "join-bow",
                       "repo2uast", "repos2uast", "uast2prox", "source2df", "source2bow",
                       "id2vec_preproc", "id2vec_train", "id2vec_postproc", "bow2vw",
                       "enry", "bigartm", "dump"):
            sys.argv = [main.__file__, action]
            main.main()

        sys.argv = args
        argparse.ArgumentParser.error = error
        self.assertEqual(sum(handlers), len(handlers))

    def test_empty(self):
        args = sys.argv
        error = argparse.ArgumentParser.error
        argparse.ArgumentParser.error = lambda self, message: None

        sys.argv = [main.__file__]
        with captured_output() as (stdout, _, _):
            main.main()

        sys.argv = args
        argparse.ArgumentParser.error = error
        self.assertIn("usage:", stdout.getvalue())

if __name__ == "__main__":
    unittest.main()
