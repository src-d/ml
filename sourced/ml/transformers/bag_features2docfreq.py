import operator

from pyspark import RDD

from sourced.ml.transformers import Transformer


class BagFeatures2DocFreq(Transformer):
    def __call__(self, rows: RDD):
        head = rows \
            .map(lambda x: x[0]) \
            .distinct() \
            .map(lambda x: (x[0], 1)) \
            .reduceByKey(operator.add)
        if self.explained:
            self._log.info("toDebugString():\n%s", head.toDebugString().decode())
        return head.collectAsMap()
