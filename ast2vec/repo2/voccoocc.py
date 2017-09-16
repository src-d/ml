import logging

from ast2vec.repo2.cooccbase import Repo2CooccBase
from ast2vec.voccoocc import VocabularyCooccurrences


class Repo2VocCoocc(Repo2CooccBase):
    """
    Converts UASTs to co-occurrence matrices which correspond to the predefined vocabulary.
    """
    MODEL_CLASS = VocabularyCooccurrences

    def __init__(self, vocabulary: dict, *args, **kwargs):
        """
        :param vocabulary: {token: index} mapping.
        """
        super().__init__(*args, **kwargs)
        self._vocabulary = vocabulary

    def _get_vocabulary(self):
        return self._vocabulary

    def _get_result(self, word2ind, mat):
        return mat

    def _update_dict(self, generator, word2ind, tokens):
        tokens.extend(generator)
