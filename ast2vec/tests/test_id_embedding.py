import argparse
import os
import tempfile
import unittest

import asdf
import numpy
from scipy.sparse import coo_matrix
import tensorflow as tf

from ast2vec import DocumentFrequencies
from ast2vec.id_embedding import preprocess
from ast2vec.model import split_strings, assemble_sparse_matrix
from ast2vec.tests.test_dump import captured_output


class IdEmbeddingTests(unittest.TestCase):
    def test_preprocess(self):
        VOCAB = 4096
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                output=tmpdir, df=os.path.join(tmpdir, "docfreq.asdf"),
                input=[os.path.join(os.path.dirname(__file__), "coocc")],
                vocabulary_size=VOCAB, shard_size=VOCAB)
            with captured_output() as (out, err, log):
                preprocess(args)
            self.assertFalse(out.getvalue())
            self.assertFalse(err.getvalue())
            self.assertIn("Skipped", log.getvalue())
            self.assertIn("error.asdf", log.getvalue())
            self.assertEqual(
                sorted(os.listdir(tmpdir)),
                ["col_sums.txt", "col_vocab.txt", "docfreq.asdf", "row_sums.txt", "row_vocab.txt",
                 "shard-000-000.pb"])
            df = DocumentFrequencies(source=os.path.join(tmpdir, "docfreq.asdf"))
            self.assertEqual(len(df), VOCAB)
            self.assertEqual(df.docs, len(os.listdir(args.input[0])) - 1)
            with open(os.path.join(tmpdir, "col_sums.txt")) as fin:
                col_sums = fin.read()
            with open(os.path.join(tmpdir, "row_sums.txt")) as fin:
                row_sums = fin.read()
            self.assertEqual(col_sums, row_sums)
            with open(os.path.join(tmpdir, "col_vocab.txt")) as fin:
                col_vocab = fin.read()
            with open(os.path.join(tmpdir, "row_vocab.txt")) as fin:
                row_vocab = fin.read()
            self.assertEqual(col_vocab, row_vocab)
            self.assertEqual(row_vocab.split("\n"), df.tokens())
            for word in row_vocab.split("\n"):
                self.assertGreater(df[word], 0)
            with open(os.path.join(tmpdir, "shard-000-000.pb"), "rb") as fin:
                features = tf.parse_single_example(
                    fin.read(),
                    features={
                        "global_row": tf.FixedLenFeature([VOCAB], dtype=tf.int64),
                        "global_col": tf.FixedLenFeature([VOCAB], dtype=tf.int64),
                        "sparse_local_row": tf.VarLenFeature(dtype=tf.int64),
                        "sparse_local_col": tf.VarLenFeature(dtype=tf.int64),
                        "sparse_value": tf.VarLenFeature(dtype=tf.float32)
                    })
            with tf.Session() as session:
                global_row, global_col, local_row, local_col, value = session.run(
                    [features[n] for n in ("global_row", "global_col", "sparse_local_row",
                                           "sparse_local_col", "sparse_value")])
            self.assertEqual(set(range(VOCAB)), set(global_row))
            self.assertEqual(set(range(VOCAB)), set(global_col))
            nnz = 1421193
            self.assertEqual(value.values.shape, (nnz,))
            self.assertEqual(local_row.values.shape, (nnz,))
            self.assertEqual(local_col.values.shape, (nnz,))
            numpy.random.seed(0)
            all_tokens = row_vocab.split("\n")
            chosen_indices = numpy.random.choice(list(range(VOCAB)), 128, replace=False)
            chosen = [all_tokens[i] for i in chosen_indices]
            freqs = numpy.zeros((len(chosen),) * 2, dtype=int)
            index = {w: i for i, w in enumerate(chosen)}
            chosen = set(chosen)
            for path in os.listdir(args.input[0]):
                with asdf.open(os.path.join(args.input[0], path)) as model:
                    if model.tree["meta"]["model"] != "co-occurrences":
                        continue
                    matrix = assemble_sparse_matrix(model.tree["matrix"]).tocsr()
                    tokens = split_strings(model.tree["tokens"])
                    interesting = {i for i, t in enumerate(tokens) if t in chosen}
                    for y in interesting:
                        row = matrix[y]
                        yi = index[tokens[y]]
                        for x, v in zip(row.indices, row.data):
                            if x in interesting:
                                freqs[yi, index[tokens[x]]] += v
            matrix = coo_matrix((
                value.values, ([global_row[row] for row in local_row.values],
                               [global_col[col] for col in local_col.values])),
                shape=(VOCAB, VOCAB))
            matrix = matrix.tocsr()[chosen_indices][:, chosen_indices].todense().astype(int)
            self.assertTrue((matrix == freqs).all())


if __name__ == "__main__":
    unittest.main()
