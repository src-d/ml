from pyspark import Row, RDD, SparkContext

from sourced.ml.algorithms import log_tf_log_idf
from sourced.ml.transformers.uast2bag_features import Uast2BagFeatures
from sourced.ml.transformers.transformer import Transformer


class TFIDF(Transformer):
    """
    Calculates TF-IDF (log-log) weight for every feature.
    """
    Columns = Uast2BagFeatures.Columns

    def __init__(self, df: dict, ndocs: int, sc: SparkContext, **kwargs):
        """
        :param df: dict containing document frequencies calculated for the given stream.
        :param ndocs: total number of documents
        :param sc: spark context used to broadcast `df
        """
        super().__init__(**kwargs)
        self.df = df
        self.ndocs = ndocs
        self.sc = sc

    def __call__(self, head: RDD):
        """

        :param head: pyspark rdd where each row is named tuple with `token`, `document` and `value`
                   (term frequency) fields. One can use Uast2TermFreq Transformer to calculate
                   such rdd.
        :return: rdd after applying TFIDF
        """
        c = self.Columns
        df = self.sc.broadcast(self.df)
        ndocs = self.ndocs
        head = head \
            .filter(lambda x: df.value.get(x[c.token]) is not None) \
            .map(lambda x: Row(**{
                c.token: x[c.token],
                c.document: x[c.document],
                c.value: log_tf_log_idf(df=df.value[x[c.token]], tf=x[c.value], ndocs=ndocs)}))
        df.unpersist(blocking=True)
        return head
