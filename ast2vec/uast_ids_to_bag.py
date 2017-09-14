from collections import defaultdict

from ast2vec.bblfsh_roles import SIMPLE_IDENTIFIER
from ast2vec.token_parser import TokenParser


class FakeVocabulary:
    def __getitem__(self, item):
        return item


class UastIds2Bag:
    """
    Converts a UAST to a bag-of-identifiers.
    """
    def __init__(self, vocabulary, token_parser=None):
        """
        :param vocabulary: The mapping from tokens to bag keys. If None, no mapping is performed.
        :param token_parser: Specify token parser if you want to use a custome one. \
            :class:'TokenParser' is used if it is not specified.
        """
        self._vocabulary = vocabulary if vocabulary is not None else FakeVocabulary()
        self._token_parser = TokenParser() if token_parser is None else token_parser

    @property
    def vocabulary(self):
        return self._vocabulary

    def uast_to_bag(self, uast, role=SIMPLE_IDENTIFIER):
        """
        Converts a UAST to a bag-of-words. The weights are identifier frequencies.
        The identifiers are preprocessed by :class:`TokenParser`.

        :param uast:
        :param role: Specify role of bblfsh Node if you want to get bag of words form them.
        :return:
        """
        stack = [uast]
        bag = defaultdict(int)
        while stack:
            node = stack.pop(0)
            if role in node.roles:
                for sub in self._token_parser.process_token(node.token):
                    try:
                        bag[self._vocabulary[sub]] += 1
                    except KeyError:
                        continue
            stack.extend(node.children)
        return bag
