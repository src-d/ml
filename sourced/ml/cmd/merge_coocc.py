import logging
import operator
from uuid import uuid4

from modelforge.progress_bar import progress_bar
import numpy as np
from scipy.sparse import coo_matrix

from sourced.ml.cmd.args import handle_input_arg
from sourced.ml.extractors.helpers import filter_kwargs
from sourced.ml.models import OrderedDocumentFrequencies, Cooccurrences
from sourced.ml.transformers import CooccModelSaver
from sourced.ml.utils.engine import pause
from sourced.ml.utils.spark import create_spark


MAX_INT32 = 2**31 - 1


@pause
def merge_coocc_entry(args):
    log = logging.getLogger("merge_coocc")
    log.setLevel(args.log_level)
    filepaths = list(handle_input_arg(args.input, log))
    log.info("Found %d files", len(filepaths))
    df = OrderedDocumentFrequencies().load(args.docfreq)
    if args.no_spark:
        merge_coocc_entry_no_spark(df, filepaths, log, args)
    else:
        merge_coocc_entry_spark(df, filepaths, log, args)


def merge_coocc_entry_spark(df, filepaths, log, args):
    session_name = "merge_coocc-%s" % uuid4()
    session = create_spark(session_name, **filter_kwargs(args.__dict__, create_spark))
    spark_context = session.sparkContext
    global_index = spark_context.broadcast(df.order)

    coocc_rdds = []
    for path in progress_bar(filepaths, log):
        coocc = Cooccurrences().load(path)
        rdd = coocc.matrix_to_rdd(spark_context)  # rdd structure: ((row, col), weight)
        tokens = spark_context.broadcast(coocc.tokens)
        coocc_rdds.append(
            rdd.map(lambda row: ((global_index.value.get(tokens.value[row[0][0]], -1),
                                  global_index.value.get(tokens.value[row[0][1]], -1)),
                                 np.uint32(row[1])))
               .filter(lambda row: row[0][0] >= 0))

    log.info("Union of concurrence matrices...")
    rdd = spark_context \
        .union(coocc_rdds) \
        .reduceByKey(lambda x, y: min(MAX_INT32, x + y))
    CooccModelSaver(args.output, df)(rdd)


def merge_coocc_entry_no_spark(df, filepaths, log, args):
    log.info("Without spark")
    shape = (len(df) + 1, len(df) + 1)
    result = coo_matrix(shape, dtype=np.uint32)
    for path in progress_bar(filepaths, log):
        coocc = Cooccurrences().load(path)
        index = [df.order.get(x, len(df) + 1) for x in coocc.tokens]
        rows = [index[x] for x in coocc.matrix.row]
        cols = [index[x] for x in coocc.matrix.col]
        result += coo_matrix(
            (coocc.matrix.data, (rows, cols)), shape=shape, dtype=np.uint32)
        indx_overflow = np.where(result.data > MAX_INT32)
        if indx_overflow[0].size > 0:
            log.warning("Overflow in %d elements."
                        "They will be saturated to %d" % (indx_overflow[0].size, MAX_INT32))
            result.data[indx_overflow] = MAX_INT32
    Cooccurrences() \
        .construct(df.tokens(), result[:-1, :-1]) \
        .save(args.output, (df,))
