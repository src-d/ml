import operator

from pyspark import Row, RDD

from sourced.ml.transformers.transformer import Transformer


class BagFeatures2TermFreq(Transformer):
    class Columns:
        """
        Stores column names for return value.
        """
        token = "token"
        document = "document"
        value = "value"

    def __call__(self, rows: RDD):
        c = self.Columns
        return rows \
            .reduceByKey(operator.add) \
            .map(lambda x: Row(**{
                c.token: x[0][0], c.document: x[0][1], c.value: x[1],
            }))
