from collections import defaultdict
from copy import deepcopy
import multiprocessing
import logging
import os

import asdf
from scipy.sparse import dok_matrix

from ast2vec.meta import generate_meta
from ast2vec.model import disassemble_sparse_matrix, merge_strings, \
    split_strings
from ast2vec.repo2base import Repo2Base, Transformer, repos2_entry, \
    ensure_bblfsh_is_running_noexc


class Repo2Coocc(Repo2Base):
    """
    Convert UAST to tuple (list of unique words, list of triplets (word1_ind,
    word2_ind, cnt)).
    """
    LOG_NAME = "repo2coocc"

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
            if self.SIMPLE_IDENTIFIER in node.roles:
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
                    self.SIMPLE_IDENTIFIER in root.roles):
            self._update_dict(self._process_token(root.token), word2ind,
                              tokens)

        for triplet in self._all2all(tokens, word2ind):
            mat[(triplet[0], triplet[1])] += triplet[2]
        return children

    def _extract_ids(self, root):
        queue = [root]
        while queue:
            node = queue.pop()
            if self.SIMPLE_IDENTIFIER in node.roles:
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


class Repo2CooccTransformer(Repo2Coocc, Transformer):
    num_processes = 2
    LOG_NAME = "repos2coocc"

    @staticmethod
    def prepare_filename(repo, output):
        """
        Remove prefixes from the repo name, so later it can be used to create
        file for each repository + replace slashes ("/") with sharps ("#").
        :param repo: name of repository
        :param output: output folder
        :return: converted repository name (removed "http://", etc.)
        """
        prefixes = ["https://", "http://"]
        for prefix in prefixes:
            if repo.startswith(prefix):
                repo_name = repo[len(prefix):]
                break
        else:
            repo_name = repo
        outfile = os.path.join(output,
                               repo_name.replace("/", "&") + ".asdf")

        return outfile

    @staticmethod
    def process_repo(url_or_path, args):
        """
        Extract vocabulary and co-occurrence matrix from repository.
        :param url_or_path: Repository URL or file system path.
        :param args: parameters to initialize Repo2Coocc: linguist,
               bblfsh_endpoint, timeout.
        :return: (list of source code identifiers, scipy.sparse co-occurrences
                 matrix)
        :rtype: tuple
        """
        obj = Repo2Coocc(**args)
        vocabulary, matrix = obj.convert_repository(url_or_path)
        return vocabulary, matrix

    @staticmethod
    def process_entry(url_or_path, args, output):
        """
        Pipeline for one repository:
        1) prepare filename
        2) extract vocabulary and co-occurrence matrix from repository
        3) save result
        :param url_or_path: Repository URL or file system path.
        :param args: parameters to initialize Repo2Coocc: linguist,
               bblfsh_endpoint, timeout.
        :param output: folder to store results
        """
        outfile = Repo2CooccTransformer.prepare_filename(url_or_path,
                                                         output)
        try:
            vocabulary, matrix = Repo2CooccTransformer.process_repo(
                url_or_path, args)
            asdf.AsdfFile({
                "tokens": merge_strings(vocabulary),
                "matrix": disassemble_sparse_matrix(matrix),
                "meta": generate_meta("co-occurrences")
            }).write_to(outfile, all_array_compression="zlib")
        except:
            log = logging.getLogger(Repo2CooccTransformer.LOG_NAME)
            log.exception(
                "Unhandled error in Repo2CooccTransformer.process_entry() "
                "at %s." % url_or_path)

    def transform(self, repos, output, num_processes=None):
        """
        Extract co-occurrence matrices & list of tokens for each repository ->
        save to output folder.
        :param repos: "repos" is the list of repository URLs or paths or \
                  files with repository URLS or paths.
        :param output: "output" is the output directory where to store the \
                        results.
        :param num_processes: number of threads to use, if negative - use all \
               CPUs.
        :return: None
        """
        if num_processes is None:
            num_processes = self.num_processes
        self.num_processes = num_processes
        if num_processes < 0:
            num_processes = multiprocessing.cpu_count()

        inputs = []

        if isinstance(repos, str):
            repos = [repos]

        for repo in repos:
            # check if it's a text file
            if os.path.isfile(repo):
                with open(repo) as f:
                    inputs.extend(l.strip() for l in f)
            else:
                inputs.append(repo)

        os.makedirs(output, exist_ok=True)

        args = self._make_args()

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(Repo2CooccTransformer.process_entry,
                         zip(inputs, [args] * len(inputs),
                             [output] * len(inputs)))

    def _make_args(self):
        endpoint = self._bblfsh[0]._channel._channel.target()
        return {"linguist": self._linguist, "bblfsh_endpoint": endpoint,
                "timeout": self._timeout}


def repo2coocc(url_or_path, linguist=None, bblfsh_endpoint=None,
               timeout=Repo2Base.DEFAULT_BBLFSH_TIMEOUT):
    """
    Extract vocabulary and co-occurrence matrix from repository.
    :param url_or_path: Repository URL or file system path.
    :param linguist: path to githib/linguist or src-d/enry.
    :param bblfsh_endpoint: Babelfish server's address.
    :param timeout: Babelfish server request timeout.
    :return: (list of source code identifiers, scipy.sparse co-occurrences
             matrix)
    :rtype: tuple
    """
    return Repo2CooccTransformer.process_repo(
        url_or_path, {"linguist": linguist,
                      "bblfsh_endpoint": bblfsh_endpoint,
                      "timeout": timeout})


def repo2coocc_entry(args):
    ensure_bblfsh_is_running_noexc()
    vocabulary, matrix = repo2coocc(
        args.repository, linguist=args.linguist, bblfsh_endpoint=args.bblfsh,
        timeout=args.timeout)
    asdf.AsdfFile({
        "tokens": merge_strings(vocabulary),
        "matrix": disassemble_sparse_matrix(matrix),
        "meta": generate_meta("co-occurrences")
    }).write_to(args.output, all_array_compression="zlib")


def repos2coocc_process(repo, args):
    log = logging.getLogger("repos2coocc")
    args_ = deepcopy(args)

    outfile = Repo2CooccTransformer.prepare_filename(repo, args.output)

    args_.output = outfile
    args_.repository = repo
    try:
        repo2coocc_entry(args_)
    except:
        log.exception("Unhandled error in repo2coocc_entry().")


def repos2coocc_entry(args):
    return repos2_entry(args, repos2coocc_process)


def print_coocc(tree, dependencies):
    words = split_strings(tree["tokens"])
    m_shape = tree["matrix"]["shape"]
    nnz = tree["matrix"]["data"][0].shape[0]

    print("Number of words:", len(words))
    print("First 10 words:", words[:10])
    print("Matrix:", ", shape:", m_shape, "number of non zero elements", nnz)
