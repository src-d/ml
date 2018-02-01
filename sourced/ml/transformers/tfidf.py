from collections import namedtuple
import typing

from pyspark import Row, RDD

from sourced.ml.algorithms import log_tf_log_idf
from sourced.ml.extractors import BagsExtractor
from sourced.ml.transformers.transformer import Transformer


class TFIDF(Transformer):
    """
    Converts a single UAST into the weighted set (dictionary), where elements are strings
    and the values are floats. The derived classes must implement uast_to_bag().
    """
    class Columns:
        """
        Stores column names for return value.
        """
        token = "token"
        document = "document"
        value = "value"

    def __init__(self, tf: RDD, df: RDD, docfreq_threshold: typing.Optional[int]=None, **kwargs):
        """
        :param tf: pyspark rdd where each row is named tuple with `token`, `document` and `value` \
                   (term frequency) fields. One can use Uast2TermFreq Transformer to calculate \
                   such rdd.
        :param df: pyspark rdd where each row is named tuple with `token` and `value` (document \
                   frequency) fields. One can use Uast2TermFreq Transformer to calculate such rdd.
        :param docfreq_threshold: The minimum number of occurrences of an element to be included \
                                  into the bag
        :param weight: TF-IDF will be multiplied by this weight to change importance of specific \
                       bag extractor
        """
        super().__init__(**kwargs)

        self.tf = tf
        self.df = df
        self.docfreq_threshold = docfreq_threshold if docfreq_threshold is not None \
            else BagsExtractor.DEFAULT_DOCFREQ_THRESHOLD

    @property
    def docfreq_threhold(self):
        return self._docfreq_threshold

    @docfreq_threhold.setter
    def docfreq_threshold(self, value):
        if not isinstance(value, int):
            raise TypeError("docfreq_threshold must be an integer, got %s" % type(value))
        if value < 1:
            raise ValueError("docfreq_threshold must be >= 1, got %d" % value)
        self._docfreq_threshold = value

    def _get_log_name(self):
        return type(self).__name__

    def __call__(self, head):
        ndocs = self.df.count()
        docfreq_threhold = self.docfreq_threhold
        df_pruned = self.df.filter(lambda x: x.value >= docfreq_threhold)
        Columns = self.Columns
        return self.tf \
            .map(lambda x: (x.token, (x.document, x.value))) \
            .join(df_pruned) \
            .map(lambda x: Row(**{Columns.token: x[0],  # token
                                  Columns.document: x[1][0][0],  # document identifier
                                  Columns.value: log_tf_log_idf(
                                                   df=x[1][1],  # document frequency
                                                   tf=x[1][0][1],  # term frequency
                                                   ndocs=ndocs)}))
