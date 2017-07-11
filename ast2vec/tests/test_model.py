import datetime
import os
import pickle
import unittest

import numpy
from scipy.sparse import csr_matrix

from ast2vec.dump import GenericModel
import ast2vec.model
from ast2vec.model import merge_strings, split_strings, \
    assemble_sparse_matrix, disassemble_sparse_matrix, Model
import ast2vec.tests.models as paths
from ast2vec.tests.fake_requests import FakeRequests


class Model1(Model):
    def _load(self, tree):
        pass


class Model2(Model):
    NAME = "model2"

    def _load(self, tree):
        pass


def get_path(name):
    return os.path.join(os.path.dirname(__file__), name)


class ModelTests(unittest.TestCase):
    def test_file(self):
        model = GenericModel(source=get_path(paths.ID2VEC))
        self._validate_meta(model)

    def test_id(self):
        def route(url):
            if url.endswith(GenericModel.INDEX_FILE):
                return '{"models": {"id2vec": {' \
                       '"92609e70-f79c-46b5-8419-55726e873cfc": ' \
                       '{"url": "https://xxx"}}}}'.encode()
            self.assertEqual("https://xxx", url)
            with open(get_path(paths.ID2VEC), "rb") as fin:
                return fin.read()

        ast2vec.model.requests = FakeRequests(route)
        model = GenericModel(source="92609e70-f79c-46b5-8419-55726e873cfc")
        self._validate_meta(model)

    def test_url(self):
        def route(url):
            self.assertEqual("https://xxx", url)
            with open(get_path(paths.ID2VEC), "rb") as fin:
                return fin.read()

        ast2vec.model.requests = FakeRequests(route)
        model = GenericModel(source="https://xxx")
        self._validate_meta(model)

    def test_auto(self):
        class FakeModel(GenericModel):
            NAME = "id2vec"

        def route(url):
            if url.endswith(GenericModel.INDEX_FILE):
                return '{"models": {"id2vec": {' \
                       '"92609e70-f79c-46b5-8419-55726e873cfc": ' \
                       '{"url": "https://xxx"}, ' \
                       '"default": "92609e70-f79c-46b5-8419-55726e873cfc"' \
                       '}}}'.encode()
            self.assertEqual("https://xxx", url)
            with open(get_path(paths.ID2VEC), "rb") as fin:
                return fin.read()

        ast2vec.model.requests = FakeRequests(route)
        model = FakeModel()
        self._validate_meta(model)

    def test_init_with_model(self):
        model1 = Model1(source=get_path(paths.ID2VEC))
        # init with correct model
        Model1(source=model1)
        # init with wrong model
        with self.assertRaises(TypeError):
            Model2(source=model1)

    def _validate_meta(self, model):
        meta = model.meta
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta, {
            'created_at': datetime.datetime(2017, 6, 18, 17, 37, 6, 255615),
            'dependencies': [],
            'model': 'id2vec',
            'uuid': '92609e70-f79c-46b5-8419-55726e873cfc',
            'version': [1, 0, 0]})


class SerializationTests(unittest.TestCase):
    def test_merge_strings(self):
        strings = ["a", "bc", "def"]
        merged = merge_strings(strings)
        self.assertIsInstance(merged, dict)
        self.assertIn("strings", merged)
        self.assertIn("lengths", merged)
        self.assertIsInstance(merged["strings"], numpy.ndarray)
        self.assertEqual(merged["strings"].shape, (1,))
        self.assertEqual(merged["strings"][0], b"abcdef")
        self.assertIsInstance(merged["lengths"], numpy.ndarray)
        self.assertEqual(merged["lengths"].shape, (3,))
        self.assertEqual(merged["lengths"][0], 1)
        self.assertEqual(merged["lengths"][1], 2)
        self.assertEqual(merged["lengths"][2], 3)

    def test_split_strings(self):
        strings = split_strings({
            "strings": numpy.array([b"abcdef"]),
            "lengths": numpy.array([1, 2, 3])
        })
        self.assertEqual(strings, ["a", "bc", "def"])

    def test_disassemble_sparse_matrix(self):
        arr = numpy.zeros((10, 10), dtype=numpy.float32)
        numpy.random.seed(0)
        arr[numpy.random.randint(0, 10, (50, 2))] = 1
        mat = csr_matrix(arr)
        dis = disassemble_sparse_matrix(mat)
        self.assertIsInstance(dis, dict)
        self.assertIn("shape", dis)
        self.assertIn("format", dis)
        self.assertIn("data", dis)
        self.assertEqual(dis["shape"], arr.shape)
        self.assertEqual(dis["format"], "csr")
        self.assertIsInstance(dis["data"], (tuple, list))
        self.assertEqual(len(dis["data"]), 3)
        self.assertTrue((dis["data"][0] == mat.data).all())
        self.assertTrue((dis["data"][1] == mat.indices).all())
        self.assertTrue((dis["data"][2] == mat.indptr).all())

    def test_assemble_sparse_matrix(self):
        tree = {
            "shape": (3, 10),
            "format": "csr",
            "data": [numpy.arange(1, 8),
                     numpy.array([0, 4, 1, 5, 2, 3, 8]),
                     numpy.array([0, 2, 4, 7])]
        }
        mat = assemble_sparse_matrix(tree)
        self.assertIsInstance(mat, csr_matrix)
        self.assertTrue((mat.data == tree["data"][0]).all())
        self.assertTrue((mat.indices == tree["data"][1]).all())
        self.assertTrue((mat.indptr == tree["data"][2]).all())
        self.assertEqual(mat.shape, (3, 10))
        self.assertEqual(mat.dtype, numpy.int)

    def test_pickle(self):
        id2vec = GenericModel(source=get_path(paths.ID2VEC))
        id2vec.tree = {"meta": id2vec.tree["meta"]}
        res = pickle.dumps(id2vec)
        id2vec_rec = pickle.loads(res)

        for k in id2vec.__dict__:
            self.assertEqual(id2vec.__dict__[k], id2vec_rec.__dict__[k])


if __name__ == "__main__":
    unittest.main()
