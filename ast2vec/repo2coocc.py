from collections import defaultdict

from scipy.sparse import dok_matrix

from ast2vec.coocc import Cooccurrences
from ast2vec.meta import generate_meta
from ast2vec.model import disassemble_sparse_matrix, merge_strings
from ast2vec.repo2base import Repo2Base, RepoTransformer, repos2_entry, repo2_entry
from ast2vec.bblfsh_roles import SIMPLE_IDENTIFIER


class Repo2Coocc(Repo2Base):
    """
    Convert UAST to tuple (list of unique words, list of triplets (word1_ind,
    word2_ind, cnt)).
    """
    MODEL_CLASS = Cooccurrences

    def convert_uasts(self, uast_generator):
        word2ind = dict()
        dok_mat = defaultdict(int)
        for uast in uast_generator:
            self._traverse_uast(uast.uast, word2ind, dok_mat)

        n_tokens = len(word2ind)
        mat = dok_matrix((n_tokens, n_tokens))

        if n_tokens == 0:
            return [], mat.tocoo()

        for coord in dok_mat:
            mat[coord[0], coord[1]] = dok_mat[coord]

        words = [p[1] for p in sorted((word2ind[w], w) for w in word2ind)]
        return words, mat.tocoo()

    def _flatten_children(self, root):
        ids = []
        stack = list(root.children)
        for node in stack:
            if SIMPLE_IDENTIFIER in node.roles:
                ids.append(node)
            else:
                stack.extend(node.children)
        return ids

    @staticmethod
    def _update_dict(generator, word2ind, tokens):
        for token in generator:
            word2ind.setdefault(token, len(word2ind))
            tokens.append(token)

    @staticmethod
    def _all2all(words, word2ind):
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                wi = word2ind[words[i]]
                wj = word2ind[words[j]]
                yield wi, wj, 1
                yield wj, wi, 1

    def _process_node(self, root, word2ind, mat):
        children = self._flatten_children(root)

        tokens = []
        for ch in children:
            self._update_dict(self._process_token(ch.token), word2ind, tokens)

        if (root.token.strip() is not None and root.token.strip() != "" and
                SIMPLE_IDENTIFIER in root.roles):
            self._update_dict(self._process_token(root.token), word2ind,
                              tokens)

        for triplet in self._all2all(tokens, word2ind):
            mat[(triplet[0], triplet[1])] += triplet[2]
        return children

    def _extract_ids(self, root):
        queue = [root]
        while queue:
            node = queue.pop()
            if SIMPLE_IDENTIFIER in node.roles:
                yield node.token
            for child in node.children:
                queue.append(child)

    def _traverse_uast(self, root, word2ind, dok_mat):
        """
        Traverses UAST and extract the co-occurrence matrix.
        """
        stack = [root]
        new_stack = []

        while stack:
            for node in stack:
                children = self._process_node(node, word2ind, dok_mat)
                new_stack.extend(children)
            stack = new_stack
            new_stack = []


class Repo2CooccTransformer(RepoTransformer):
    WORKER_CLASS = Repo2Coocc

    @classmethod
    def result_to_tree(cls, result):
        vocabulary, matrix = result
        if not vocabulary:
            raise ValueError("Empty vocabulary")
        return {
            "tokens": merge_strings(vocabulary),
            "matrix": disassemble_sparse_matrix(matrix),
            "meta": generate_meta(cls.WORKER_CLASS.MODEL_CLASS.NAME)
        }


def repo2coocc_entry(args):
    return repo2_entry(args, Repo2CooccTransformer)


def repos2coocc_entry(args):
    return repos2_entry(args, Repo2CooccTransformer)
