from collections import defaultdict
import logging
import os
import sys

import asdf
import numpy
from scipy.sparse import dok_matrix
import tensorflow as tf

from ast2vec.meta import generate_meta
from ast2vec.model import merge_strings, split_strings, assemble_sparse_matrix
import ast2vec.swivel as swivel
from ast2vec.repo2nbow import Repo2nBOW


def preprocess(args):
    log = logging.getLogger("preproc")
    log.info("Scanning the inputs...")
    inputs = []
    for i in args.input:
        if os.path.isdir(i):
            inputs.extend([os.path.join(i, f) for f in os.listdir(i)])
        else:
            inputs.append(i)
    log.info("Reading word indices from %d files...", len(inputs))
    all_words = defaultdict(int)
    for i, path in enumerate(inputs):
        sys.stdout.write("%d / %d\r" % (i + 1, len(inputs)))
        for w in split_strings(asdf.open(path).tree["tokens"]):
            all_words[w] += 1
    vs = int(args.vocabulary_size)
    if len(all_words) < vs:
        vs = len(all_words)
    sz = int(args.shard_size)
    vs -= vs % sz
    log.info("Effective vocabulary size: %d", vs)
    log.info("Truncating the vocabulary...")
    words = numpy.array(list(all_words.keys()))
    freqs = numpy.array(list(all_words.values()), dtype=numpy.int64)
    del all_words
    chosen_indices = numpy.argpartition(
        freqs, len(freqs) - vs)[len(freqs) - vs:]
    chosen_freqs = freqs[chosen_indices]
    chosen_words = words[chosen_indices]
    del words
    del freqs
    log.info("Sorting the vocabulary...")
    sorted_indices = numpy.argsort(-chosen_freqs)
    chosen_freqs = chosen_freqs[sorted_indices]
    chosen_words = chosen_words[sorted_indices]
    word_indices = {w: i for i, w in enumerate(chosen_words)}
    if args.df is not None:
        log.info("Writing the document frequencies to %s...", args.df)
        asdf.AsdfFile({
            "tokens": merge_strings(chosen_words),
            "freqs": chosen_freqs,
            "meta": generate_meta("docfreq")
        }).write_to(args.df, all_array_compression="zlib")
    del chosen_freqs
    del chosen_words
    log.info("Combining individual co-occurrence matrices...")
    ccmatrix = dok_matrix((vs, vs), dtype=numpy.int64)
    for i, path in enumerate(inputs):
        sys.stdout.write("%d / %d\r" % (i + 1, len(inputs)))
        with open(path, "rb") as fin:
            tree = asdf.open(path).tree
            words = split_strings(tree["tokens"])
            indices = []
            mapped_indices = []
            for i, w in enumerate(words):
                gi = word_indices.get(w)
                if gi is not None:
                    indices.append(i)
                    mapped_indices.append(gi)
            matrix = assemble_sparse_matrix(tree["matrix"]).tocsr() \
                [indices, indices]
            for ri, rs, rf in zip(mapped_indices, matrix.indptr,
                                  matrix.indptr[1:]):
                for ii, v in zip(matrix.indices[rs:rf], matrix.data[rs:rf]):
                    ccmatrix[ri, mapped_indices[ii]] += v
    log.info("Planning the sharding...")
    bool_sums = ccmatrix.indptr[1:] - ccmatrix.indptr[:-1]
    reorder = numpy.argsort(-bool_sums)
    log.info("Writing the shards...")
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
    log.info("Success")


def run_swivel(args):
    swivel.FLAGS = args
    swivel.main(args)


def postprocess(args):
    log = logging.getLogger("postproc")
    log.info("Parsing the embeddings at %s...", args.swivel_output_directory)
    tokens = []
    embeddings = []
    swd = args.swivel_output_directory
    with open(os.path.join(swd, "row_embedding.tsv")) as frow:
        with open(os.path.join(swd, "col_embedding.tsv")) as fcol:
            for i, (lrow, lcol) in enumerate(zip(frow, fcol)):
                if i % 10000 == (10000 - 1):
                    sys.stdout.write("%d\r" % (i + 1))
                prow, pcol = (l.split("\t", 1) for l in (lrow, lcol))
                assert prow[0] == pcol[0]
                tokens.append(prow[0][:Repo2nBOW.MAX_TOKEN_LENGTH])
                erow, ecol = \
                    (numpy.fromstring(p[1], dtype=numpy.float32, sep="\t")
                     for p in (prow, pcol))
                embeddings.append((erow + ecol) / 2)
    print(" " * 20 + "\r")
    log.info("Generating numpy arrays...")
    embeddings = numpy.array(embeddings, dtype=numpy.float32)
    tokens = numpy.array(tokens, dtype=str)
    log.info("Writing %s...", args.result)
    asdf.AsdfFile({
        "tokens": merge_strings(tokens),
        "embeddings": embeddings,
        "meta": generate_meta("id2vec")
    }).write_to(args.result, all_array_compression="zlib")
