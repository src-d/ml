import numpy

from ast2vec.repo2base import Repo2Base


class Repo2Coocc(Repo2Base):
    """
    Convert UAST to tuple (list of unique words, list of triplets (word1_ind,
    word2_ind, cnt))
    """
    LOG_NAME = "repo2coo_matrix"

    def convert_uasts(self, uast_generator):
        word2ind = dict()
        dok_matrix = dict()
        for uast in uast_generator:
            wi = dict.setdefault(w, len(dict))
            pass

    def flatten_children(self, root):
        ids = []
        stack = list(root.children)
        for node in stack:
            if self.SIMPLE_IDENTIFIER in node.roles:
                ids.append(node)
            else:
                stack.extend(node.children)
        return ids

    def tokenizer(self, token):
        return list(self._process_token(token))

    @staticmethod
    def update_dict(words, word2ind):
        for w in words:
            _ = word2ind.setdefault(w, len(word2ind))
            # if w not in word2ind:
            #     word2ind[w] = len(word2ind)

    @staticmethod
    def all2all(words, word2ind):
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                wi = word2ind.setdefault(words[i], len(word2ind))
                wj = word2ind.setdefault(words[j], len(word2ind))
                yield (wi, wj, 1)
                yield (wj, wi, 1)

    def process_node(self, root, word2ind, mat):
        children = self.flatten_children(root)

        tokens = []
        for ch in children:
            t = self.tokenizer(ch.token)
            self.update_dict(t, word2ind)
            tokens.extend(t)
        if (root.token.strip() is not None and root.token.strip() != '' and
                    self.SIMPLE_IDENTIFIER in root.roles):
            t = self.tokenizer(root.token)
            self.update_dict(t, word2ind)
            tokens.extend(t)
        for triplet in self.all2all(tokens, word2ind):
            mat[(triplet[0], triplet[1])] += triplet[2]
        return children

    def extract_ids(self, root):
        if self.SIMPLE_IDENTIFIER in root.roles:
            yield root.token
        for child in root.children:
            for r in self.extract_ids(child):
                yield r

    def traverse_uast(self, root, word2ind, dok_matrix):
        # Travers UAST and extract dependencies
        stack = [root]
        new_stack = []

        while stack:
            for node in stack:
                children = self.process_node(node, word2ind, dok_matrix)
                new_stack.extend(children)
            stack = new_stack
            new_stack = []
        # words = [p[1] for p in sorted([(word2ind[w], w) for w in word2ind],
        #                               key=itemgetter(0))]
        # return words, dok_matrix.tocoo()


def repo2coocc(url_or_path, linguist=None, bblfsh_endpoint=None):
    obj = Repo2Coocc(linguist=linguist, bblfsh_endpoint=bblfsh_endpoint)
    vocabulary, matrix = obj.convert_repository(url_or_path)
    return vocabulary, matrix


def repo2coocc_entry(args):
    vocabulary, matrix = repo2coocc(args.repository, linguist=args.linguist,
                                    bblfsh_endpoint=args.bblfsh)
    numpy.savez_compressed(args.output, tokens=vocabulary, matrix=matrix)


