import logging
import os
import shutil
import warnings

import numpy
try:
    import tensorflow as tf
except ImportError as e:
    warnings.warn("Tensorflow is not installed, dependent functionality is unavailable.")

from modelforge.progress_bar import progress_bar
from sourced.ml.algorithms.id_embedding import extract_coocc_matrix
from sourced.ml.models import Cooccurrences, DocumentFrequencies


def _int64s(xs):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(xs)))


def _floats(xs):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=list(xs)))


def id2vec_preprocess(args):
    """
    Loads co-occurrence matrices for several repositories and generates the
    document frequencies and the Swivel protobuf dataset.

    :param args: :class:`argparse.Namespace` with "input", "vocabulary_size", \
                 "shard_size", "df" and "output".
    :return: None
    """
    log = logging.getLogger("preproc")
    log.info("Loading docfreq model from %s", args.docfreq_in)
    df_model = DocumentFrequencies(log_level=args.log_level).load(source=args.docfreq_in)
    coocc_model = Cooccurrences().load(args.input)
    if numpy.any(coocc_model.matrix.data < 0):
        raise ValueError(("Co-occurrence matrix %s contains negative elements. "
                          "Please check its correctness.") % args.input)
    if numpy.any(numpy.isnan(coocc_model.matrix.data)):
        raise ValueError(("Co-occurrence matrix %s contains nan elements. "
                          "Please check its correctness.") % args.input)

    try:
        df_meta = coocc_model.get_dep(DocumentFrequencies.NAME)
        if df_model.meta != df_meta:
            raise ValueError((
                "Document frequency model you provided does not match dependency inside "
                "Cooccurrences model:\nargs.docfreq.meta:\n%s\ncoocc_model.get_dep"
                "(\"docfreq\")\n%s\n") % (df_model.meta, df_meta))
    except KeyError:
        pass  # There is no docfreq dependency

    vs = args.vocabulary_size
    if len(df_model) < vs:
        vs = len(df_model)
    sz = args.shard_size
    if vs < sz:
        raise ValueError(
            "vocabulary_size=%s is less than shard_size=%s. You should specify a smaller "
            "shard_size (e.g. shard_size=%s)." % (vs, sz, vs))
    vs -= vs % sz
    log.info("Effective vocabulary size: %d", vs)
    df_model = df_model.greatest(vs)
    log.info("Sorting the vocabulary...")
    chosen_words = sorted(df_model.tokens())
    word_indices = {w: i for i, w in enumerate(chosen_words)}

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, "row_vocab.txt"), "w") as out:
        out.write('\n'.join(chosen_words))
    log.info("Saved row_vocab.txt")
    shutil.copyfile(os.path.join(args.output, "row_vocab.txt"),
                    os.path.join(args.output, "col_vocab.txt"))
    log.info("Saved col_vocab.txt")
    del chosen_words

    ccmatrix = extract_coocc_matrix((vs, vs), word_indices, coocc_model)

    log.info("Planning the sharding...")
    bool_sums = ccmatrix.indptr[1:] - ccmatrix.indptr[:-1]
    reorder = numpy.argsort(-bool_sums)
    with open(os.path.join(args.output, "row_sums.txt"), "w") as out:
        out.write('\n'.join(map(str, bool_sums.tolist())))
    log.info("Saved row_sums.txt")
    shutil.copyfile(os.path.join(args.output, "row_sums.txt"),
                    os.path.join(args.output, "col_sums.txt"))
    log.info("Saved col_sums.txt")

    log.info("Writing the shards...")
    os.makedirs(args.output, exist_ok=True)
    nshards = vs // args.shard_size
    for row in progress_bar(range(nshards), log, expected_size=nshards):
        for col in range(nshards):
            indices_row = reorder[row::nshards]
            indices_col = reorder[col::nshards]
            shard = ccmatrix[indices_row][:, indices_col].tocoo()

            example = tf.train.Example(features=tf.train.Features(feature={
                "global_row": _int64s(indices_row),
                "global_col": _int64s(indices_col),
                "sparse_local_row": _int64s(shard.row),
                "sparse_local_col": _int64s(shard.col),
                "sparse_value": _floats(shard.data)}))

            with open(os.path.join(args.output,
                                   "shard-%03d-%03d.pb" % (row, col)),
                      "wb") as out:
                out.write(example.SerializeToString())
    log.info("Success")
