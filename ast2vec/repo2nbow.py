from collections import defaultdict
from copy import deepcopy
import logging
import math
import os

import asdf

from ast2vec.meta import generate_meta, ARRAY_COMPRESSION
from ast2vec.id2vec import Id2Vec
from ast2vec.df import DocumentFrequencies
from ast2vec.repo2base import Repo2Base, Transformer, repos2_entry, \
    ensure_bblfsh_is_running_noexc


class Repo2nBOW(Repo2Base):
    """
    Implements the step repository -> :class:`ast2vec.nbow.NBOW`.
    """
    LOG_NAME = "repo2nbow"

    def __init__(self, id2vec, docfreq, tempdir=None, linguist=None,
                 log_level=logging.INFO, bblfsh_endpoint=None,
                 timeout=Repo2Base.DEFAULT_BBLFSH_TIMEOUT):
        super(Repo2nBOW, self).__init__(
            tempdir=tempdir, linguist=linguist, log_level=log_level,
            bblfsh_endpoint=bblfsh_endpoint, timeout=timeout)
        self._id2vec = id2vec
        self._docfreq = docfreq

    @property
    def id2vec(self):
        return self._id2vec

    @property
    def docfreq(self):
        return self._docfreq

    def convert_uasts(self, uast_generator):
        freqs = defaultdict(int)
        for uast in uast_generator:
            bag = self._uast_to_bag(uast.uast)
            for key, freq in bag.items():
                freqs[key] += freq
        missing = []
        vocabulary = self._id2vec.tokens
        for key, val in freqs.items():
            try:
                freqs[key] = math.log(1 + val) * math.log(
                    self._docfreq.docs / self._docfreq[vocabulary[key]])
            except KeyError:
                missing.append(key)
        for key in missing:
            del freqs[key]
        return dict(freqs)

    def _uast_to_bag(self, uast):
        stack = [uast]
        bag = defaultdict(int)
        while stack:
            node = stack.pop(0)
            if self.SIMPLE_IDENTIFIER in node.roles:
                for sub in self._process_token(node.token):
                    try:
                        bag[self._id2vec[sub]] += 1
                    except KeyError:
                        pass
            stack.extend(node.children)
        return bag


class Repo2nBOWTransformer(Transformer):
    """
    Wrap the step: repository -> :class:`ast2vec.nbow.NBOW`.
    """
    LOG_NAME = "repo2nbow_transformer"

    def __init__(self, id2vec=None, docfreq=None, linguist=None,
                 gcs_bucket=None, bblfsh_endpoint=None,
                 timeout=Repo2Base.DEFAULT_BBLFSH_TIMEOUT):
        self.repo2nbow = self.init_repo2nbow(id2vec=id2vec, linguist=linguist,
                                             docfreq=docfreq, timeout=timeout,
                                             bblfsh_endpoint=bblfsh_endpoint,
                                             gcs_bucket=gcs_bucket)


    @staticmethod
    def init_repo2nbow(id2vec=None, docfreq=None, linguist=None,
              bblfsh_endpoint=None, timeout=Repo2Base.DEFAULT_BBLFSH_TIMEOUT,
              gcs_bucket=None):
        """
        Performs the step repository -> :class:`ast2vec.nbow.NBOW`.

        :param url_or_path: Repository URL or file system path.
        :param id2vec: :class:`ast2vec.Id2Vec` model.
        :param docfreq: :class:`ast2vec.DocumentFrequencies` model.
        :param linguist: path to githib/linguist or src-d/enry.
        :param bblfsh_endpoint: Babelfish server's address.
        :param timeout: Babelfish server request timeout.
        :param gcs_bucket: GCS bucket name where the models are stored.
        :return: Repo2nBOW instance initialized with id2vec & df & etc.
        :rtype: Repo2nBOW
        """
        id2vec = Id2Vec(id2vec or None, gcs_bucket=gcs_bucket)
        docfreq = DocumentFrequencies(docfreq or None, gcs_bucket=gcs_bucket)

        obj = Repo2nBOW(id2vec, docfreq, linguist=linguist,
                        bblfsh_endpoint=bblfsh_endpoint, timeout=timeout)

        return obj

    def _process_repo(self, repo, outfile):
        nbow = self.repo2nbow.convert_repository(repo)
        asdf.AsdfFile({
            "nbow": nbow,
            "meta": generate_meta("nbow", self.repo2nbow.id2vec,
                                  self.repo2nbow.docfreq)
        }).write_to(outfile, all_array_compression="zlib")

    def transform(self, X, output):
        if isinstance(X, str):
            # write result to file output
            outfile = output
            self._process_repo(X, outfile)
        elif isinstance(X, list):
            # write files to folder output
            os.makedirs(output, exist_ok=True)
            for repo in X:
                outfile = os.path.join(output, repo.replace("/", "#"))
                self._process_repo(repo, outfile)


def repo2nbow(url_or_path, id2vec=None, df=None, linguist=None,
              bblfsh_endpoint=None, timeout=Repo2Base.DEFAULT_BBLFSH_TIMEOUT,
              gcs_bucket=None):
    """
    Performs the step repository -> :class:`ast2vec.nbow.NBOW`.

    :param url_or_path: Repository URL or file system path.
    :param id2vec: :class:`ast2vec.Id2Vec` model.
    :param df: :class:`ast2vec.DocumentFrequencies` model.
    :param linguist: path to githib/linguist or src-d/enry.
    :param bblfsh_endpoint: Babelfish server's address.
    :param timeout: Babelfish server request timeout.
    :param gcs_bucket: GCS bucket name where the models are stored.
    :return: {token: weight}
    :rtype: dict
    """
    if id2vec is None:
        id2vec = Id2Vec(gcs_bucket=gcs_bucket)
    if df is None:
        df = DocumentFrequencies(gcs_bucket=gcs_bucket)
    obj = Repo2nBOW(id2vec, df, linguist=linguist,
                    bblfsh_endpoint=bblfsh_endpoint, timeout=timeout)
    nbow = obj.convert_repository(url_or_path)
    return nbow


def repo2nbow_entry(args):
    ensure_bblfsh_is_running_noexc()
    id2vec = Id2Vec(args.id2vec or None, gcs_bucket=args.gcs)
    df = DocumentFrequencies(args.df or None, gcs_bucket=args.gcs)
    linguist = args.linguist or None
    nbow = repo2nbow(args.repository, id2vec=id2vec, df=df, linguist=linguist,
                     bblfsh_endpoint=args.bblfsh, timeout=args.timeout,
                     gcs_bucket=args.gcs)
    logging.getLogger("repo2nbow").info("Writing %s...", args.output)
    asdf.AsdfFile({
        "nbow": nbow,
        "meta": generate_meta("nbow", id2vec, df)
    }).write_to(args.output, all_array_compression=ARRAY_COMPRESSION)


def repos2nbow_process(repo, args):
    log = logging.getLogger("repos2nbow")
    args_ = deepcopy(args)
    outfile = os.path.join(args.output, repo.replace("/", "#"))
    args_.output = outfile
    args_.repository = repo
    try:
        repo2nbow_entry(args_)
    except:
        log.exception("Unhandled error in repo2nbow_entry().")


def repos2nbow_entry(args):
    return repos2_entry(args, repos2nbow_process)
