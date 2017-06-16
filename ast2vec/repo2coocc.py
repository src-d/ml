from collections import defaultdict
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import Pool

from operator import itemgetter
from os.path import isfile, exists, join
from os import makedirs

from bblfsh.launcher import ensure_bblfsh_is_running
import numpy
from scipy.sparse import dok_matrix

from ast2vec.repo2base import Repo2Base


class Repo2Coocc(Repo2Base):
    """
    Convert UAST to tuple (list of unique words, list of triplets (word1_ind,
    word2_ind, cnt))
    """
    LOG_NAME = "repo2coocc"
    uasts = []

    def convert_uasts(self, uast_generator):
        # self.uasts.extend(uast_generator)
        # # return self.uasts
        # self.word2ind = word2ind = dict()
        word2ind = dict()
        # self.reference_dict = set()
        dok_mat = defaultdict(int)
        for uast in uast_generator:
            # bag = self._uast_to_bag(uast)
            # self.reference_dict.update(bag)
            # if {'contain', 'result', 'sock', 'socket'}.intersection(bag):
            #     print("?", file)
            # if  {'find', 'packag'}.intersection(bag):
            #     print("!", file)
            # if 'src/bblfsh/setup.py' == file:
            #     print("dump src/bblfsh/setup.py")
            #     print(sorted(bag))
            #     from datetime import datetime
            #     with open("setup_debug" + str(datetime.now()) + '.txt',
            #               'w') as f:
            #         f.write(str(uast))
            # if 'src/bblfsh/bblfsh/launcher.py' == file:
            #     print("dump src/bblfsh/bblfsh/launcher.py")
            #     print(sorted(bag))
            #     from datetime import datetime
            #     with open("launcher_debug" + str(datetime.now()) + '.txt', 'w') as f:
            #         f.write(str(uast))
            if uast is not None:
                self.traverse_uast(uast, word2ind, dok_mat)
            # else:
            #     print('skipped!')
            #     print('-' * 20)
            # print('number of unique words:', len(word2ind))

        n_tokens = len(word2ind)
        mat = dok_matrix((n_tokens, n_tokens))

        if n_tokens == 0:
            return [], mat

        for coord in dok_mat:
            mat[coord[0], coord[1]] = dok_mat[coord]

        words = [p[1] for p in sorted([(word2ind[w], w) for w in word2ind],
                                      key=itemgetter(0))]
        # print(words, len(words), mat)
        return words, mat.tocoo()

    def _uast_to_bag(self, uast):
        stack = [uast]
        bag = defaultdict(int)
        while stack:
            node = stack.pop(0)
            if self.SIMPLE_IDENTIFIER in node.roles:
                for sub in self._process_token(node.token):
                    bag[sub] += 1
            stack.extend(node.children)
        return bag

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

    @staticmethod
    def all2all(words, word2ind):
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                # wi = word2ind.setdefault(words[i], len(word2ind))
                wi = word2ind[words[i]]
                wj = word2ind[words[j]]
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

    def traverse_uast(self, root, word2ind, dok_mat):
        # Travers UAST and extract co occurrence matrix
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
    # print(sorted(obj.word2ind))
    # print()
    # print(len(obj.reference_dict))
    # print(sorted(obj.reference_dict))
    return vocabulary, matrix


def repo2coocc_entry(args):
    ensure_bblfsh_is_running()
    vocabulary, matrix = repo2coocc(args.repository, linguist=args.linguist,
                                    bblfsh_endpoint=args.bblfsh)
    numpy.savez_compressed(args.output, tokens=vocabulary, matrix=matrix)


def __pool_f__(v):
    repo, args, outdir = v
    args_ = deepcopy(args)
    outfile = join(outdir, repo.replace('/', '#'))
    args_.output = outfile

    args_.repository = repo
    repo2coocc_entry(args_)


def repos2coocc_entry(args):
    ensure_bblfsh_is_running()
    inputs = []

    i = args.input
    # check if it's text file
    if isfile(i):
        with open(i, 'r') as f:
            inputs.extend([l.strip() for l in f.readlines()])
    else:
        inputs.append(i)

    outdir = args.output
    if not exists(outdir):
        makedirs(outdir)

    # for input in inputs:
    #     __pool_f__((input, args, outdir))

    with Pool(processes=mp.cpu_count()) as pool:
        pool.map(__pool_f__, zip(inputs, [args] * len(inputs),
                             [outdir] * len(inputs)))

