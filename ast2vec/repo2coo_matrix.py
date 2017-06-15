from ast2vec.repo2base import Repo2Base


class Repo2CooMatrix(Repo2Base):
    """
    Convert UAST to tuple (list of unique words, list of triplets (word1_ind, word2_ind, cnt)) 
    """
    LOG_NAME = "repo2coo_matrix"

    def convert_uasts(self, uast_generator):
        for uast in uast_generator:
            wi = dict.setdefault(w, len(dict))
            pass

