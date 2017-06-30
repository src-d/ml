import shutil
from collections import defaultdict
import logging
import os
import sys

import asdf
from clint.textui import progress
import numpy
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix
import tensorflow as tf

from ast2vec.meta import generate_meta
from ast2vec.model import merge_strings, split_strings, assemble_sparse_matrix
import ast2vec.swivel as swivel
from ast2vec.repo2base import Transformer, DictAttr
from ast2vec.repo2nbow import Repo2nBOW


class PreprocessTransformer(Transformer):
    vocabulary_size = 1 << 17
    shard_size = 4096

    def __init__(self, vocabulary_size=None, shard_size=None):
        if vocabulary_size is not None:
            self.vocabulary_size = vocabulary_size
        if shard_size is not None:
            self.shard_size = shard_size

    def transform(self, X, output, df=None, vocabulary_size=None,
                  shard_size=None):
        if vocabulary_size is not None:
            self.vocabulary_size = vocabulary_size
        if shard_size is not None:
            self.shard_size = shard_size

        if isinstance(X, str):
            X = [X]

        d = {'vocabulary_size': self.vocabulary_size, 'input': X, 'df': df,
             'shard_size': self.shard_size, 'output': output}

        args = DictAttr(d)
        preprocess(args)


def preprocess(args):
    """
    Loads co-occurrence matrices for several repositories and generates the
    document frequencies and the Swivel protobuf dataset.

    :param args: :class:`argparse.Namespace` with "input", "vocabulary_size", \
                 "shard_size", "df" and "output".
    :return: None
    """
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
    for i, path in progress.bar(enumerate(inputs), expected_size=len(inputs)):
        for w in split_strings(asdf.open(path).tree["tokens"]):
            all_words[w] += 1
    vs = args.vocabulary_size
    if len(all_words) < vs:
        vs = len(all_words)
    sz = args.shard_size
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
            "docs": len(inputs),
            "meta": generate_meta("docfreq")
        }).write_to(args.df, all_array_compression="zlib")
    del chosen_freqs

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, "row_vocab.txt"), "w") as out:
        out.write('\n'.join(chosen_words))
    log.info("Saved row_vocab.txt...")
    shutil.copyfile(os.path.join(args.output, "row_vocab.txt"),
                    os.path.join(args.output, "col_vocab.txt"))
    log.info("Saved col_vocab.txt...")

    del chosen_words
    log.info("Combining individual co-occurrence matrices...")
    ccmatrix = dok_matrix((vs, vs), dtype=numpy.int64)
    for i, path in progress.bar(enumerate(inputs), expected_size=len(inputs)):
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

            matrix = csr_matrix(assemble_sparse_matrix(tree["matrix"])
                                .tocsr()[indices, indices])
            for ri, rs, rf in zip(mapped_indices, matrix.indptr,
                                  matrix.indptr[1:]):
                for ii, v in zip(matrix.indices[rs:rf], matrix.data[rs:rf]):
                    ccmatrix[ri, mapped_indices[ii]] += v
    log.info("Planning the sharding...")
    ccmatrix = ccmatrix.tocsr()
    bool_sums = ccmatrix.indptr[1:] - ccmatrix.indptr[:-1]
    with open(os.path.join(args.output, "row_sums.txt"), "w") as out:
        out.write('\n'.join(map(str, bool_sums.tolist())))
    log.info("Saved row_sums.txt...")
    shutil.copyfile(os.path.join(args.output, "row_sums.txt"),
                    os.path.join(args.output, "col_sums.txt"))
    log.info("Saved col_sums.txt...")
    reorder = numpy.argsort(-bool_sums)
    log.info("Writing the shards...")
    os.makedirs(args.output, exist_ok=True)
    nshards = vs // args.shard_size
    for row in progress.bar(range(nshards), expected_size=nshards):
        for col in range(nshards):
            def _int64s(xs):
                return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(xs)))

            def _floats(xs):
                return tf.train.Feature(
                    float_list=tf.train.FloatList(value=list(xs)))

            indices_row = reorder[row::nshards]
            indices_col = reorder[col::nshards]
            shard = coo_matrix(ccmatrix[indices_row, indices_col])

            example = tf.train.Example(features=tf.train.Features(feature={
                "global_row": _int64s(row + nshards * i for i in range(sz)),
                "global_col": _int64s(col + nshards * i for i in range(sz)),
                "sparse_local_row": _int64s(shard.row),
                "sparse_local_col": _int64s(shard.col),
                "sparse_value": _floats(shard.data)}))

            with open(os.path.join(args.output,
                                   "shard-%03d-%03d.pb" % (row, col)),
                      "wb") as out:
                out.write(example.SerializeToString())
    log.info("Success")


class SwivelTransformer(Transformer):
    FLAGS = swivel.FLAGS

    def transform(self, input_base_path=None, output_base_path=None,
                  embedding_size=None, trainable_bias=None,
                  submatrix_rows=None, submatrix_cols=None,
                  loss_multiplier=None, confidence_exponent=None,
                  confidence_scale=None, confidence_base=None,
                  learning_rate=None, optimizer=None,
                  num_concurrent_steps=None, num_readers=None, num_epochs=None,
                  per_process_gpu_memory_fraction=None, num_gpus=None,
                  logs=None):
        if input_base_path is not None:
            self.FLAGS.input_base_path = input_base_path
        if output_base_path is not None:
            self.FLAGS.output_base_path = output_base_path
        if embedding_size is not None:
            self.FLAGS.embedding_size = embedding_size
        if trainable_bias is not None:
            self.FLAGS.trainable_bias = trainable_bias
        if submatrix_rows is not None:
            self.FLAGS.submatrix_rows = submatrix_rows
        if submatrix_cols is not None:
            self.FLAGS.submatrix_cols = submatrix_cols
        if loss_multiplier is not None:
            self.FLAGS.loss_multiplier = loss_multiplier
        if confidence_exponent is not None:
            self.FLAGS.confidence_exponent = confidence_exponent
        if confidence_scale is not None:
            self.FLAGS.confidence_scale = confidence_scale
        if confidence_base is not None:
            self.FLAGS.confidence_base = confidence_base
        if learning_rate is not None:
            self.FLAGS.learning_rate = learning_rate
        if optimizer is not None:
            self.FLAGS.optimizer = optimizer
        if num_concurrent_steps is not None:
            self.FLAGS.num_concurrent_steps = num_concurrent_steps
        if num_readers is not None:
            self.FLAGS.num_readers = num_readers
        if num_epochs is not None:
            self.FLAGS.num_epochs = num_epochs
        if per_process_gpu_memory_fraction is not None:
            self.FLAGS.per_process_gpu_memory_fraction = \
                per_process_gpu_memory_fraction
        if num_gpus is not None:
            self.FLAGS.num_gpus = num_gpus
        if logs is not None:
            self.FLAGS.logs = logs

        run_swivel(self.FLAGS)


def run_swivel(args):
    """
    Trains the Swivel model. Wraps swivel.py, adapted from
    https://github.com/vmarkovtsev/models/blob/master/swivel/swivel.py

    :param args: :class:`argparse.Namespace` identical to \
                 :class:`tf.app.flags`.
    :return: None
    """
    swivel.FLAGS = args
    logging.getLogger("tensorflow").handlers.clear()
    swivel.main(args)


class PostprocessTransformer(Transformer):
    def transform(self, X, output):
        """
        Merges row and column embeddings produced by Swivel and writes the
        Id2Vec model.

        :param X: folder that contains files after swivel training. The files \
                  are read from this folder and the model is written to the\
                  'output'.
        :param output: file to store results
        :return: None
        """
        swivel_output_directory = X
        result = output
        args = DictAttr({"swivel_output_directory": swivel_output_directory,
                         "result": result})

        postprocess(args)



def postprocess(args):
    """
    Merges row and column embeddings produced by Swivel and writes the Id2Vec
    model.

    :param args: :class:`argparse.Namespace` with "swivel_output_directory" \
                 and "result". The text files are read from \
                 `swivel_output_directory` and the model is written to \
                 `result`.
    :return: None
    """
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
                    sys.stdout.flush()
                prow, pcol = (l.split("\t", 1) for l in (lrow, lcol))
                assert prow[0] == pcol[0]
                tokens.append(prow[0][:Repo2nBOW.MAX_TOKEN_LENGTH])
                erow, ecol = \
                    (numpy.fromstring(p[1], dtype=numpy.float32, sep="\t")
                     for p in (prow, pcol))
                embeddings.append((erow + ecol) / 2)
    log.info("Generating numpy arrays...")
    embeddings = numpy.array(embeddings, dtype=numpy.float32)
    tokens = numpy.array(tokens, dtype=str)
    log.info("Writing %s...", args.result)
    asdf.AsdfFile({
        "tokens": merge_strings(tokens),
        "embeddings": embeddings,
        "meta": generate_meta("id2vec")
    }).write_to(args.result, all_array_compression="zlib")
