import itertools
import operator

from scipy import sparse
from bblfsh import Node
from pyspark import Row
from pyspark.rdd import PipelinedRDD

from sourced.ml.models import Cooccurrences, OrderedDocumentFrequencies
from sourced.ml.transformers import Transformer
from sourced.ml.utils import bblfsh_roles, EngineConstants


class CooccModelSaver(Transformer):
    def __init__(self, output, df_model: OrderedDocumentFrequencies, **kwargs):
        super().__init__(**kwargs)
        self.tokens_list = df_model.tokens()
        self.output = output
        self.df_model = df_model

    def __call__(self, sparse_matrix: PipelinedRDD):
        """
        Saves Cooccurrences asdf model to disk.

        :param sparse_matrix: rdd with 3 columns: matrix row, matrix column,  cell value. Use
            :class:`.CooccConstructor` to construct RDD from uasts.
        :return:
        """
        rows = sparse_matrix.collect()

        mat_index, mat_weights = zip(*rows)
        mat_row, mat_col = zip(*mat_index)
        tokens_num = len(self.tokens_list)

        self._log.info("Building matrix...")
        matrix = sparse.coo_matrix((mat_weights, (mat_row, mat_col)),
                                   shape=(tokens_num, tokens_num))
        Cooccurrences() \
            .construct(self.tokens_list, matrix) \
            .save(self.output, deps=(self.df_model,))


class CooccConstructor(Transformer):
    """
    Co-occurrence matrix calculation transformer.
    You can find an algorithm full description in blog.sourced.tech/posts/id2vec.
    """
    def __init__(self, token2index, token_parser, namespace="", **kwargs):
        super().__init__(**kwargs)
        self.token2index = token2index
        self.token_parser = token_parser
        self.namespace = namespace

    def _flatten_children(self, root):
        ids = []
        stack = list(root.children)
        while stack:
            node = stack.pop(0)
            if bblfsh_roles.IDENTIFIER in node.roles:
                ids.append(node)
            else:
                stack.extend(node.children)
        return ids

    def _traverse_uast(self, uast: Node):
        stack = [uast]

        while stack:
            node = stack.pop(0)
            children = self._flatten_children(node)
            tokens = [token for ch in children for token in self.token_parser(ch.token)]

            if bblfsh_roles.IDENTIFIER in node.roles:
                tokens.extend(self.token_parser(node.token))
            for pair in itertools.permutations(tokens, 2):
                yield pair

            stack.extend(children)

    def __call__(self, uasts):
        return uasts \
            .flatMap(self._process_row) \
            .reduceByKey(operator.add)

    def _process_row(self, row: Row):
        for uast in row[EngineConstants.Columns.Uast]:
            for token1, token2 in self._traverse_uast(uast):
                try:
                    yield (self.token2index.value[self.namespace + token1],
                           self.token2index.value[self.namespace + token2]), 1
                except KeyError:
                    # Do not have token1 or token2 in the token2index map
                    pass
