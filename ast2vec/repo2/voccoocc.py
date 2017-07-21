import logging

from ast2vec.repo2.cooccbase import Repo2CooccBase
from ast2vec.voccoocc import VocabularyCooccurrences


class Repo2VocCoocc(Repo2CooccBase):
    """
    Converts UASTs to co-occurrence matrices which correspond to the predefined vocabulary.
    """
    MODEL_CLASS = VocabularyCooccurrences

    def __init__(self, vocabulary: dict, tempdir=None, linguist=None,
                 log_level=logging.INFO, bblfsh_endpoint=None,
                 timeout=Repo2CooccBase.DEFAULT_BBLFSH_TIMEOUT):
        """
        :param vocabulary: {token: index} mapping.
        """
        super(Repo2VocCoocc, self).__init__(
            tempdir=tempdir, linguist=linguist, log_level=log_level,
            bblfsh_endpoint=bblfsh_endpoint, timeout=timeout)
        self._vocabulary = vocabulary

    def _get_vocabulary(self):
        return self._vocabulary

    def _get_result(self, word2ind, mat):
        return mat

    def _update_dict(self, generator, word2ind, tokens):
        tokens.extend(generator)
