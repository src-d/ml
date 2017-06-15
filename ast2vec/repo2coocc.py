import numpy

from ast2vec.repo2base import Repo2Base


class Repo2Coocc(Repo2Base):
    """
    Convert UAST to tuple (list of unique words, list of triplets (word1_ind,
    word2_ind, cnt))
    """
    LOG_NAME = "repo2coo_matrix"

    def convert_uasts(self, uast_generator):
        for uast in uast_generator:
            wi = dict.setdefault(w, len(dict))
            pass


def repo2coocc(url_or_path, linguist=None, bblfsh_endpoint=None):
    obj = Repo2Coocc(linguist=linguist, bblfsh_endpoint=bblfsh_endpoint)
    vocabulary, matrix = obj.convert_repository(url_or_path)
    return vocabulary, matrix


def repo2coocc_entry(args):
    vocabulary, matrix = repo2coocc(args.repository, linguist=args.linguist,
                                    bblfsh_endpoint=args.bblfsh)
    numpy.savez_compressed(args.output, tokens=vocabulary, matrix=matrix)
