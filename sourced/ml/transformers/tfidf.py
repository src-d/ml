from pyspark import Row, RDD

from sourced.ml.algorithms import log_tf_log_idf
from sourced.ml.models import DocumentFrequencies
from sourced.ml.transformers.uast2bag_features import Uast2BagFeatures
from sourced.ml.transformers.transformer import Transformer


class TFIDF(Transformer):
    """
    Calculates TF-IDF (log-log) weight for every feature.
    """
    Columns = Uast2BagFeatures.Columns

    def __init__(self, df: DocumentFrequencies, **kwargs):
        """
        :param tf: pyspark rdd where each row is named tuple with `token`, `document` and `value` \
                   (term frequency) fields. One can use Uast2TermFreq Transformer to calculate \
                   such rdd.
        :param df: DocumentFrequencies model calculated for the given `tf` stream.
        """
        super().__init__(**kwargs)
        self.df = df

    def __call__(self, head):
        c = self.Columns
        df = self.df
        return head \
            .filter(lambda x: df.get(x[c.token]) is not None) \
            .map(lambda x: Row(**{
                c.token: x[c.token],
                c.document: x[c.document],
                c.value: log_tf_log_idf(df=df[x[c.token]], tf=x[c.value], ndocs=df.docs)}))
