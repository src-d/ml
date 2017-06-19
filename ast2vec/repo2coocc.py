from collections import defaultdict
from copy import deepcopy
import logging
import os

import asdf
from scipy.sparse import dok_matrix

from ast2vec.meta import generate_meta
from ast2vec.model import disassemble_sparse_matrix, merge_strings
from ast2vec.repo2base import Repo2Base, repos2_entry, \
    ensure_bblfsh_is_running_noexc


class Repo2Coocc(Repo2Base):
    """
    Convert UAST to tuple (list of unique words, list of triplets (word1_ind,
    word2_ind, cnt))
    """
    LOG_NAME = "repo2coocc"

    def convert_uasts(self, uast_generator):
        word2ind = dict()
        dok_mat = defaultdict(int)
        for uast in uast_generator:
            self.traverse_uast(uast.uast, word2ind, dok_mat)

        n_tokens = len(word2ind)
        mat = dok_matrix((n_tokens, n_tokens))

        if n_tokens == 0:
            return [], mat.tocoo()

        for coord in dok_mat:
            mat[coord[0], coord[1]] = dok_mat[coord]

        words = [p[1] for p in sorted((word2ind[w], w) for w in word2ind)]
        return words, mat.tocoo()

    def flatten_children(self, root):
        ids = []
        stack = list(root.children)
        for node in stack:
            if self.SIMPLE_IDENTIFIER in node.roles:
                ids.append(node)
            else:
                stack.extend(node.children)
        return ids

    @staticmethod
    def update_dict(generator, word2ind, tokens):
        for token in generator:
            word2ind.setdefault(token, len(word2ind))
            tokens.append(token)

    @staticmethod
    def all2all(words, word2ind):
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                wi = word2ind[words[i]]
                wj = word2ind[words[j]]
                yield wi, wj, 1
                yield wj, wi, 1

    def process_node(self, root, word2ind, mat):
        children = self.flatten_children(root)

        tokens = []
        for ch in children:
            self.update_dict(self._process_token(ch.token), word2ind, tokens)

        if (root.token.strip() is not None and root.token.strip() != "" and
                    self.SIMPLE_IDENTIFIER in root.roles):
            self.update_dict(self._process_token(root.token), word2ind,
                             tokens)

        for triplet in self.all2all(tokens, word2ind):
            mat[(triplet[0], triplet[1])] += triplet[2]
        return children

    def extract_ids(self, root):
        if self.SIMPLE_IDENTIFIER in root.roles:
            yield root.token
        for child in root.children:
            for r in self.extract_ids(child):
                yield r

    def traverse_uast(self, root, word2ind, dok_mat):
        """
        Traverse UAST and extract the co-occurence matrix
        """
        stack = [root]
        new_stack = []

        while stack:
            for node in stack:
                children = self.process_node(node, word2ind, dok_mat)
                new_stack.extend(children)
            stack = new_stack
            new_stack = []


def repo2coocc(url_or_path, linguist=None, bblfsh_endpoint=None):
    obj = Repo2Coocc(linguist=linguist, bblfsh_endpoint=bblfsh_endpoint)
    vocabulary, matrix = obj.convert_repository(url_or_path)
    return vocabulary, matrix


def repo2coocc_entry(args):
    ensure_bblfsh_is_running_noexc()
    vocabulary, matrix = repo2coocc(args.repository, linguist=args.linguist,
                                    bblfsh_endpoint=args.bblfsh)
    asdf.AsdfFile({
        "tokens": merge_strings(vocabulary),
        "matrix": disassemble_sparse_matrix(matrix),
        "meta": generate_meta("co-occurrences")
    }).write_to(args.output, all_array_compression="zlib")


def repos2coocc_process(repo, args):
    log = logging.getLogger("repos2coocc")
    args_ = deepcopy(args)
    outfile = os.path.join(args.output, repo.replace("/", "#"))
    args_.output = outfile
    args_.repository = repo
    try:
        repo2coocc_entry(args_)
    except:
        log.exception("Unhandled error in repo2coocc_entry().")


def repos2coocc_entry(args):
    return repos2_entry(args, repos2coocc_process)
