from collections import defaultdict
import os
import pickle
import sys

import numpy
from scipy.sparse import dok_matrix
import tensorflow as tf

import ast2vec.swivel as swivel


def preprocess(args):
    print("Reading word indices from %d files..." % len(args.input))
    all_words = defaultdict(int)
    for i, path in enumerate(args.input):
        sys.stdout.write("%d / %d\r" % (i + 1, len(args.input)))
        with open(path, "rb") as fin:
            words = pickle.load(fin)
            for w in words:
                all_words[w] += 1
    vs = args.vocabulary_size
    if len(all_words) < vs:
        vs = len(all_words)
    sz = args.shard_size
    vs -= vs % sz
    print("Effective vocabulary size:", vs)
    print("Truncating the vocabulary...")
    words = numpy.array(list(all_words.keys()))
    freqs = numpy.array(list(all_words.values()), dtype=numpy.int64)
    del all_words
    chosen_indices = numpy.argpartition(
        freqs, len(freqs) - vs)[len(freqs) - vs:]
    chosen_freqs = freqs[chosen_indices]
    chosen_words = words[chosen_indices]
    del words
    del freqs
    print("Sorting the vocabulary...")
    sorted_indices = numpy.argsort(-chosen_freqs)
    chosen_freqs = chosen_freqs[sorted_indices]
    chosen_words = chosen_words[sorted_indices]
    word_indices = {w: i for i, w in enumerate(chosen_words)}
    if args.df is not None:
        print("Writing the document frequencies to %s..." % args.df)
        with open(args.df, "w") as fout:
            for p in zip(chosen_words, chosen_freqs):
                fout.write("%s\t%d\n" % p)
    del chosen_freqs
    del chosen_words
    print("Combining individual co-occurrence matrices...")
    ccmatrix = dok_matrix((vs, vs), dtype=numpy.int64)
    for i, path in enumerate(args.input):
        sys.stdout.write("%d / %d\r" % (i + 1, len(args.input)))
        with open(path, "rb") as fin:
            words = pickle.load(fin)
            indices = []
            mapped_indices = []
            for i, w in enumerate(words):
                gi = word_indices.get(w)
                if gi is not None:
                    indices.append(i)
                    mapped_indices.append(gi)
            matrix = pickle.load(fin)[indices, indices].tocsr()
            for ri, rs, rf in zip(mapped_indices, matrix.indptr,
                                  matrix.indptr[1:]):
                for ii, v in zip(matrix.indices[rs:rf], matrix.data[rs:rf]):
                    ccmatrix[ri, mapped_indices[ii]] += v
    print("Planning the sharding...")
    bool_sums = ccmatrix.indptr[1:] - ccmatrix.indptr[:-1]
    reorder = numpy.argsort(-bool_sums)
    print("Writing the shards...")
    nshards = vs / args.shard_size
    for row in range(nshards):
        for col in range(nshards):
            sys.stdout.write(
                "%d / %d\r" % (row * nshards + col, nshards * nshards))

            def _int64s(xs):
                return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(xs)))

            def _floats(xs):
                return tf.train.Feature(
                    float_list=tf.train.FloatList(value=list(xs)))

            indices_row = reorder[row::nshards]
            indices_col = reorder[col::nshards]
            shard = ccmatrix[indices_row, indices_col].tocoo()

            example = tf.train.Example(features=tf.train.Features(feature={
                "global_row": _int64s(row + nshards * i for i in range(sz)),
                "global_col": _int64s(col + nshards * i for i in range(sz)),
                "sparse_local_row": _int64s(shard.row),
                "sparse_local_col": _int64s(shard.col),
                "sparse_value": _floats(shard.data)}))

            with open(os.path.join(args.output,
                                   "shard-%03d-%03d.pb" % (row, col)),
                      "w") as out:
                out.write(example.SerializeToString())
    print(" " * 80 + "\r")


def run_swivel(args):
    swivel.FLAGS = args
    swivel.main(args)


def postprocess(args):
    print("Parsing the embeddings at %s..." % args.swivel_output_directory)
    tokens = []
    embeddings = []
    swd = args.swivel_output_directory
    with open(os.path.join(swd, "row_embedding.tsv")) as frow:
        with open(os.path.join(swd, "col_embedding.tsv")) as fcol:
            for lrow, lcol in zip(frow, fcol):
                prow, pcol = (l.split("\t", 1) for l in (lrow, lcol))
                assert prow[0] == pcol[0]
                tokens.append(prow[0])
                erow, ecol = \
                    (numpy.fromstring(p[1], dtype=numpy.float32, sep="\t")
                     for p in (prow, pcol))
                embeddings.append((erow + ecol) / 2)
    embeddings = numpy.array(embeddings)
    print("Writing %s..." % args.npz)
    numpy.savez_compressed(args.npz, embeddings=embeddings, tokens=tokens)
