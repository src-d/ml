import os
import tempfile
import unittest
import numpy as np
from scipy.sparse import csc_matrix

from sourced.ml.models import BOW
from sourced.ml.models.model_converters.merge_bow import MergeBOW


class MergeBOWTests(unittest.TestCase):
    def setUp(self):
        self.model1 = BOW() \
            .construct(["doc_1", "doc_2", "doc_3"], ["f.tok_1", "k.tok_2", "f.tok_3"],
                       csc_matrix((np.array([1, 2]), (np.array([0, 1]), np.array([1, 0]))),
                                  shape=(3, 3)))
        self.model1._meta = {"dependencies": [{"model": "docfreq", "uuid": "uuid"}]}
        self.model2 = BOW() \
            .construct(["doc_4", "doc_5", "doc_6"], ["f.tok_1", "k.tok_2", "f.tok_3"],
                       csc_matrix((np.array([3, 4]), (np.array([0, 1]), np.array([1, 0]))),
                                  shape=(3, 3)))
        self.model2._meta = {"dependencies": [{"model": "docfreq", "uuid": "uuid"}]}
        self.merge_results = [[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 3, 0], [4, 0, 0], [0, 0, 0]]
        self.merge_bow = MergeBOW()

    def test_convert_model_base(self):
        self.merge_bow.convert_model(self.model1)
        self.assertListEqual(self.merge_bow.documents, ["doc_1", "doc_2", "doc_3"])
        self.assertListEqual(self.merge_bow.tokens, ["f.tok_1", "k.tok_2", "f.tok_3"])
        for i, row in enumerate(self.merge_bow.matrix[0].toarray()):
            self.assertListEqual(list(row), self.merge_results[i])
        self.assertEqual(self.merge_bow.deps, [{'uuid': 'uuid', 'model': 'docfreq'}])
        self.merge_bow.convert_model(self.model2)
        self.assertListEqual(self.merge_bow.documents,
                             ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"])
        self.assertListEqual(self.merge_bow.tokens, ["f.tok_1", "k.tok_2", "f.tok_3"])
        for i, arr in enumerate(self.merge_bow.matrix):
            for j, row in enumerate(arr.toarray()):
                self.assertListEqual(list(row), self.merge_results[i * 3 + j])
        self.assertEqual(self.merge_bow.deps, [{"model": "docfreq", "uuid": "uuid"}])

    def test_convert_model_error(self):
        self.merge_bow.convert_model(self.model1)
        self.model2._tokens = ["f.tok_1", "k.tok_2"]
        with self.assertRaises(ValueError):
            self.merge_bow.convert_model(self.model2)
        self.model2._tokens = ["f.tok_1", "k.tok_2", "f.tok_3", "f.tok_4"]
        with self.assertRaises(ValueError):
            self.merge_bow.convert_model(self.model2)

    def test_finalize_base(self):
        self.merge_bow.convert_model(self.model1)
        self.merge_bow.convert_model(self.model2)
        with tempfile.TemporaryDirectory(prefix="merge-bow-") as tmpdir:
            dest = os.path.join(tmpdir, "bow.asdf")
            self.merge_bow.finalize(0, dest)
            bow = BOW().load(dest)
            self.assertListEqual(bow.documents,
                                 ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"])
            self.assertListEqual(bow.tokens, ["f.tok_1", "k.tok_2", "f.tok_3"])
            for i, row in enumerate(bow.matrix.toarray()):
                self.assertListEqual(list(row), self.merge_results[i])
            self.assertEqual(bow.meta["dependencies"], [{'uuid': 'uuid', 'model': 'docfreq'}])

    def test_finalize_reduce(self):
        self.merge_bow.convert_model(self.model1)
        self.merge_bow.features_namespaces = "f."
        with tempfile.TemporaryDirectory(prefix="merge-bow-") as tmpdir:
            dest = os.path.join(tmpdir, "bow.asdf")
            self.merge_bow.finalize(0, dest)
            bow = BOW().load(dest)
            self.assertListEqual(bow.documents, ["doc_1", "doc_2", "doc_3"])
            self.assertListEqual(bow.tokens, ["f.tok_1", "f.tok_3"])
            for i, row in enumerate(bow.matrix.toarray()):
                self.assertListEqual(list(row), self.merge_results[i][::2])
            self.assertEqual(bow.meta["dependencies"], [{'uuid': 'uuid', 'model': 'docfreq'}])

    def test_save_path(self):
        self.assertEqual(self.merge_bow._save_path(0, "bow.asdf"), "bow.asdf")
        self.assertEqual(self.merge_bow._save_path(0, "bow"), os.path.join("bow", "bow_0.asdf"))


if __name__ == '__main__':
    unittest.main()
