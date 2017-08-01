import tempfile
import unittest

import asdf
from google.protobuf.message import DecodeError

import ast2vec.tests as tests
import ast2vec.tests.models as paths
import ast2vec.resolve_symlink
from ast2vec import Source, Repo2SourceTransformer, Repo2Base
from ast2vec import resolve_symlink
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

    def default_source_model(self, tmpdir):
        r2cc = Repo2SourceTransformer(timeout=50, linguist=tests.ENRY)
        r2cc.transform(DATA_DIR_SOURCE, output=tmpdir, num_processes=1)
        path_of_result = Repo2SourceTransformer.prepare_filename(DATA_DIR_SOURCE, tmpdir)

        validate_asdf_file(self, path_of_result)
        model = Source(source=path_of_result)
        return model

    def test_obj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.default_source_model(tmpdir)
            true_model = Source(source=paths.SOURCE)
            self.assertEqual(len(model.sources), 1)
            self.assertEqual(len(model.uasts), 1)
            self.assertEqual(true_model.sources[0], model.sources[0])
            self.assertEqual(true_model.uasts[0], model.uasts[0])

    def check_empty_source_model(self, model):
        self.assertEqual(model.filenames, [])
        self.assertEqual(model.uasts, [])
        self.assertEqual(model.sources, [])

    def test_SymlinkToNotExistingFile(self):
        save_resolve_symlink = ast2vec.resolve_symlink.resolve_symlink

        def resolve_symlink_raise(_):
            raise resolve_symlink.SymlinkToNotExistingFile('error')

        ast2vec.resolve_symlink.resolve_symlink = resolve_symlink_raise

        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.default_source_model(tmpdir)
            self.check_empty_source_model(model)
        ast2vec.resolve_symlink.resolve_symlink = save_resolve_symlink

    def test_reach_max_size_file_limit(self):
        save_MAX_FILE_SIZE = Repo2Base.MAX_FILE_SIZE
        Repo2Base.MAX_FILE_SIZE = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.default_source_model(tmpdir)
            self.check_empty_source_model(model)
            Repo2Base.MAX_FILE_SIZE = save_MAX_FILE_SIZE

    def test_bblfsh_parse_return_none(self):
        def bblfsh_parse_return_none(_, __, ___):
            return None

        save_bblfsh_parse = Repo2Base._bblfsh_parse
        Repo2Base._bblfsh_parse = bblfsh_parse_return_none
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.default_source_model(tmpdir)
            self.check_empty_source_model(model)
            Repo2Base._bblfsh_parse = save_bblfsh_parse

    def test_bblfsh_parse_raise_DecodeError(self):
        def bblfsh_parse_return_none(_, __, ___):
            raise DecodeError()

        save_bblfsh_parse = Repo2Base._bblfsh_parse
        Repo2Base._bblfsh_parse = bblfsh_parse_return_none
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.default_source_model(tmpdir)
            self.check_empty_source_model(model)
            Repo2Base._bblfsh_parse = save_bblfsh_parse


if __name__ == "__main__":
    unittest.main()
