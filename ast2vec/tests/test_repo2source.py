import unittest
import tempfile

import asdf

import ast2vec.tests as tests
import ast2vec.tests.models as paths
from ast2vec import Source, Repo2SourceTransformer
from ast2vec.tests.models import DATA_DIR_SOURCE


def validate_asdf_file(obj, filename):
    data = asdf.open(filename)
    obj.assertIn("meta", data.tree)
    obj.assertIn("sources", data.tree)
    obj.assertIn("uasts", data.tree)
    obj.assertEqual(0, len(data.tree["meta"]["dependencies"]))
    obj.assertEqual(data.tree["meta"]["model"], "source")


class Repo2SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_obj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r2cc = Repo2SourceTransformer(timeout=50, linguist=tests.ENRY)
            r2cc.transform(DATA_DIR_SOURCE, output=tmpdir, num_processes=1)
            path_of_result = Repo2SourceTransformer.prepare_filename(DATA_DIR_SOURCE, tmpdir)
            true_path = paths.SOURCE
            validate_asdf_file(self, path_of_result)
            model = Source(source=path_of_result)
            true_model = Source(source=true_path)
            self.assertEqual(len(model.sources), 1)
            self.assertEqual(len(model.uasts), 1)
            self.assertEqual(true_model.sources[0], model.sources[0])
            self.assertEqual(true_model.uasts[0], model.uasts[0])


if __name__ == "__main__":
    unittest.main()
