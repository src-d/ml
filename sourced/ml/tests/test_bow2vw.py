import argparse
import logging
import tempfile
import unittest

import sourced
import sourced.ml.tests.models as paths
from sourced.ml.models import BOW
from sourced.ml.cmd import bow2vw


class Bow2vwTests(unittest.TestCase):
    def test_convert_bow_to_vw(self):
        bow = BOW().load(source=paths.BOW)
        vocabulary = ["i.", "i.*", "i.Activity", "i.AdapterView", "i.ArrayAdapter", "i.Arrays"]
        with tempfile.NamedTemporaryFile(prefix="sourced.ml-vw-") as fout:
            logging.getLogger().level = logging.ERROR
            try:
                bow.convert_bow_to_vw(fout.name)
            finally:
                logging.getLogger().level = logging.INFO
            fout.seek(0)
            contents = fout.read().decode()
        hits = 0
        for word in vocabulary:
            if " %s:" % word in contents:
                hits += 1
        self.assertEqual(hits, 4)

    def test_repo2bow(self):
        called = [None] * 2

        def fake_convert_bow_to_vw(*args):
            called[:] = args

        args = argparse.Namespace(bow=paths.BOW, id2vec=paths.ID2VEC, output="out.test")
        backup = sourced.ml.models.BOW.convert_bow_to_vw
        sourced.ml.models.BOW.convert_bow_to_vw = fake_convert_bow_to_vw
        try:
            bow2vw(args)
        finally:
            sourced.ml.models.BOW.convert_bow_to_vw = backup
        self.assertIsInstance(called[0], BOW)
        self.assertEqual(called[1], "out.test")


if __name__ == "__main__":
    unittest.main()
