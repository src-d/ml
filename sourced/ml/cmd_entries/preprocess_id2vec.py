import logging
import os
import shutil

import numpy
import tensorflow as tf

from modelforge.progress_bar import progress_bar
from sourced.ml.algorithms.id_embedding import _extract_coocc_matrix
from sourced.ml.models import Cooccurrences, DocumentFrequencies


def _int64s(xs):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(xs)))


def _floats(xs):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=list(xs)))


def preprocess_id2vec(args):
    """
    Loads co-occurrence matrices for several repositories and generates the
    document frequencies and the Swivel protobuf dataset.

    :param args: :class:`argparse.Namespace` with "input", "vocabulary_size", \
                 "shard_size", "df" and "output".
    :return: None
    """
    log = logging.getLogger("preproc")
    df_model = DocumentFrequencies().load(source=args.docfreq)
    coocc_model = Cooccurrences().load(args.input)
    if coocc_model.meta['dependencies']:
        try:
            df_meta = coocc_model.get_dep("docfreq")
            if df_model.meta != df_meta:
                raise ValueError((
                    "Document frequency model you provided does not match dependency inside "
                    "Cooccurrences model:\nargs.docfreq.meta:\n%s\ncoocc_model.get_dep"
                    "(\"docfreq\")\n%s\n") % (df_model.meta, df_meta))
        except KeyError:
            pass  # There is no docfreq dependency

    word_map = df_model.docfreq
    del df_model
    vs = args.vocabulary_size
    if len(word_map) < vs:
        vs = len(word_map)
    sz = args.shard_size
    if vs < sz:
        raise ValueError(
            "vocabulary_size=%s is less than shard_size=%s. You should specify a smaller "
            "shard_size (e.g. shard_size=%s)." % (vs, sz, vs))
    vs -= vs % sz
    log.info("Effective vocabulary size: %d", vs)
    log.info("Truncating the vocabulary...")
    words = list(word_map)
    word_indices = numpy.arange(len(word_map), dtype=numpy.int32)
    freqs = numpy.fromiter(word_map.values(), numpy.int64, len(word_map))
    del word_map
    chosen_indices = numpy.argpartition(freqs, len(freqs) - vs)[len(freqs) - vs:]
    chosen_freqs = freqs[chosen_indices]
    chosen_word_indices = word_indices[chosen_indices]
    # we need to be deterministic at the cutoff frequency
    # argpartition returns random samples every time
    # so we take all words with the cutoff frequency, sort them and take the needed amount
    # finally, we replace the randomly chosen samples (border_mask) with those
    border_freq = chosen_freqs.min()
    border_mask = chosen_freqs == border_freq
    border_num = border_mask.sum()
    border_word_index_map = {words[i]: i for i in word_indices[freqs == border_freq]}
    border_words = list(border_word_index_map)
    border_words.sort()
    chosen_word_indices[border_mask] = numpy.fromiter(
        (border_word_index_map[w] for w in border_words[:border_num]), numpy.int32, border_num)
    del word_indices
    del freqs
    chosen_words = [words[i] for i in chosen_word_indices]
    del words
    del chosen_word_indices
    log.info("Sorting the vocabulary...")
    sorted_indices = numpy.argsort(chosen_words)
    chosen_freqs = chosen_freqs[sorted_indices]
    chosen_words = [chosen_words[i] for i in sorted_indices]
    del sorted_indices
    word_indices = {w: i for i, w in enumerate(chosen_words)}
    del chosen_freqs

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, "row_vocab.txt"), "w") as out:
        out.write('\n'.join(chosen_words))
    log.info("Saved row_vocab.txt")
    shutil.copyfile(os.path.join(args.output, "row_vocab.txt"),
                    os.path.join(args.output, "col_vocab.txt"))
    log.info("Saved col_vocab.txt")
    del chosen_words

    ccmatrix = _extract_coocc_matrix((vs, vs), word_indices, coocc_model)

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
