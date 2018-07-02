import logging
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
def merge_coocc(args):
    log = logging.getLogger("merge_coocc")
    log.setLevel(args.log_level)
    filepaths = list(handle_input_arg(args.input, log))
    log.info("Will merge %d files", len(filepaths))
    df = OrderedDocumentFrequencies().load(args.docfreq)
    if args.no_spark:
        merge_coocc_no_spark(df, filepaths, log, args)
    else:
        merge_coocc_spark(df, filepaths, log, args)


def load_and_check(filepaths: list, log: logging.Logger):
    """
    Load Cooccurrences models from filepaths list and perform simple check:
    1. If model contains values more than MAX_INT32 we saturate.
    2. If model contains negative values we consider it as corrupted, report and skip.
    """
    for path in progress_bar(filepaths, log):
        coocc = Cooccurrences().load(path)
        negative_values = np.where(coocc.matrix.data < 0)
        if negative_values[0].size > 0:
            log.warning("Model %s is corrupted and will be skipped. "
                        "It contains negative elements.", path)
            continue
        too_big_values = np.where(coocc.matrix.data > MAX_INT32)
        if too_big_values[0].size > 0:
            log.warning("Model %s contains elements with values more than MAX_INT32. "
                        "They will be saturated to MAX_INT32", path)
            coocc.matrix.data[too_big_values] = MAX_INT32
        yield path, coocc


def merge_coocc_spark(df, filepaths, log, args):
    session_name = "merge_coocc-%s" % uuid4()
    session = create_spark(session_name, **filter_kwargs(args.__dict__, create_spark))
    spark_context = session.sparkContext
    global_index = spark_context.broadcast(df.order)

    coocc_rdds = []

    def local_to_global(local_index):
        """
        Converts token index of co-occurrence matrix to the common index.
        For example index, 5 correspond to `get` token for a current model.
        And `get` have index 7 in the result.
        So we convert 5 to `get` via tokens list and `get` to 7 via global_index mapping.
        If global_index do not have `get` token, it returns -1.
        """
        return global_index.value.get(tokens.value[local_index], -1)

    for path, coocc in load_and_check(filepaths, log):
        rdd = coocc.matrix_to_rdd(spark_context)  # rdd structure: ((row, col), weight)
        log.info("Broadcasting tokens order for %s model...", path)
        tokens = spark_context.broadcast(coocc.tokens)
        coocc_rdds.append(
            rdd.map(lambda row: ((local_to_global(row[0][0]),
                                  local_to_global(row[0][1])),
                                 np.uint32(row[1])))
               .filter(lambda row: row[0][0] >= 0))

    log.info("Calculating the union of cooccurrence matrices...")
    rdd = spark_context \
        .union(coocc_rdds) \
        .reduceByKey(lambda x, y: min(MAX_INT32, x + y))
    CooccModelSaver(args.output, df)(rdd)


def merge_coocc_no_spark(df, filepaths, log, args):
    """
    Algorithm explanation:

    1. Although we store result in uint32, we actually never have elements greater than MAX_INT32
    2. We assume that both result and the summed matrix do not have elements greater than MAX_INT32
    3. As soon as we have a value bigger than MAX_INT32 after summing, we saturate
    4. Thus we lose 2x data range but do not allocate any additional memory and it works faster
       than MAX_UINT32 checks
    5. Only ? elements saturate in PGA so this is fine

    """
    # TODO(zurk): recheck the number of saturated elements.
    log.info("Merging cooccurrences without using PySpark")
    shape = (len(df) + 1,) * 2
    result = coo_matrix(shape, dtype=np.uint32)
    for path, coocc in load_and_check(filepaths, log):
        coocc._matrix = coo_matrix(coocc._matrix)
        index = [df.order.get(x, len(df)) for x in coocc.tokens]
        rows = [index[x] for x in coocc.matrix.row]
        cols = [index[x] for x in coocc.matrix.col]
        result += coo_matrix(
            (coocc.matrix.data, (rows, cols)), shape=shape, dtype=np.uint32)
        indx_overflow = np.where(result.data > MAX_INT32)
        if indx_overflow[0].size > 0:
            log.warning("Overflow in %d elements. They will be saturated to MAX_INT32",
                        indx_overflow[0].size)
            result.data[indx_overflow] = MAX_INT32
    Cooccurrences() \
        .construct(df.tokens(), result[:-1, :-1]) \
        .save(args.output, (df,))
