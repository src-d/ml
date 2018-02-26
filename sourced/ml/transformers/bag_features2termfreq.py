import operator

from pyspark import Row, RDD

from sourced.ml.transformers.uast2bag_features import Uast2BagFeatures
from sourced.ml.transformers.transformer import Transformer


class BagFeatures2TermFreq(Transformer):
    def __call__(self, rows: RDD):
        c = Uast2BagFeatures.Columns
        return rows \
            .reduceByKey(operator.add) \
            .map(lambda x: Row(**{
                c.token: x[0][0], c.document: x[0][1], c.value: x[1],
            }))
