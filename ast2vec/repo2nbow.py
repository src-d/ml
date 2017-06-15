from collections import defaultdict
import json
import logging
import multiprocessing
import os
from queue import Queue
import re
import shutil
import subprocess
import tempfile
import threading

from bblfsh import BblfshClient
from bblfsh.github.com.bblfsh.sdk.uast.generated_pb2 import DESCRIPTOR
import Stemmer

from ast2vec.id2vec import Id2Vec
from ast2vec.df import DocumentFrequencies


class Repo2nBOW:
    NAME_BREAKUP_RE = re.compile(r"[^a-zA-Z]+")
    STEM_THRESHOLD = 6
    SIMPLE_IDENTIFIER = DESCRIPTOR.enum_types_by_name["Role"] \
        .values_by_name["SIMPLE_IDENTIFIER"].number

    def __init__(self, id2vec, docfreq, tempdir=None, linguist=None,
                 log_level=logging.INFO, bblfsh_endpoint=None):
        self._log = logging.getLogger("repo2nbow")
        self._log.setLevel(log_level)
        self._id2vec = id2vec
        self._docfreq = docfreq
        self._stemmer = Stemmer.Stemmer("english")
        self._stemmer.maxCacheSize = 0
        self._stem_threshold = 6
        self._tempdir = tempdir
        self._linguist = "enry" if linguist is None else linguist
        self._bblfsh = [BblfshClient(bblfsh_endpoint or "0.0.0.0:9432")
                        for _ in range(multiprocessing.cpu_count())]

    def convert_repository(self, url_or_path):
        temp = not os.path.exists(url_or_path)
        if temp:
            target_dir = tempfile.mkdtemp(
                prefix="repo2nbow-", dir=self._tempdir)
            self._log.info("Cloning from %s...", url_or_path)
            try:
                subprocess.check_call(
                    ["git", "clone", "--depth=1", url_or_path, target_dir])
            except Exception as e:
                shutil.rmtree(target_dir)
                raise e from None
        else:
            target_dir = url_or_path
        try:
            self._log.info("Classifying the files...")
            classified = self._classify_files(target_dir)
            self._log.info("Fetching and processing UASTs...")

            def uast_generator():
                queue_in = Queue()
                queue_out = Queue()

                def thread_loop(thread_index):
                    while True:
                        task = queue_in.get()
                        if task is None:
                            break
                        try:
                            filename, language = task
                            uast = self._bblfsh[thread_index].parse_uast(
                                filename, language=language)
                            queue_out.put_nowait(uast)
                        except:
                            self._log.exception(
                                "Error while processing %s.", task)
                            queue_out.put_nowait(None)

                pool = [threading.Thread(target=thread_loop, args=(i,))
                        for i in range(multiprocessing.cpu_count())]
                for thread in pool:
                    thread.start()
                tasks = 0
                for lang, files in classified.items():
                    # FIXME(vmarkovtsev): remove this hardcode when https://github.com/bblfsh/server/issues/28 is resolved
                    if lang not in ("Python", "Java"):
                        continue
                    for f in files:
                        tasks += 1
                        queue_in.put_nowait((f, lang))
                for _ in pool:
                    queue_in.put_nowait(None)
                while tasks > 0:
                    yield queue_out.get()
                    tasks -= 1
                for thread in pool:
                    thread.join()

            return self.convert_uasts(uast_generator())
        finally:
            if temp:
                shutil.rmtree(target_dir)

    def convert_uasts(self, uast_generator):
        freqs = defaultdict(int)
        for uast in uast_generator:
            bag = self._uast_to_bag(uast)
            for key, freq in bag.items():
                freqs[key] += freq
        return freqs

    def _uast_to_bag(self, uast):
        stack = [uast]
        bag = defaultdict(int)
        while stack:
            node = stack.pop(0)
            if self.SIMPLE_IDENTIFIER in node.roles:
                for sub in self._process_token(node.token):
                    bag[sub] += 1
            stack.extend(node.children)

    def _classify_files(self, target_dir):
        bjson = subprocess.check_output([self._linguist, target_dir])
        classified = json.loads(bjson.decode("utf-8"))
        return classified

    def _process_token(self, token):
        for word in self._split(token):
            yield self._stem(word)

    def _stem(self, word):
        if len(word) <= self._stem_threshold:
            return word
        return self._stemmer.stemWord(word)

    @classmethod
    def _split(cls, token):
        token = token.strip()
        prev_p = [""]

        def ret(name):
            r = name.lower()
            if len(name) >= 3:
                yield r
                if prev_p[0]:
                    yield prev_p[0] + r
                    prev_p[0] = ""
            else:
                prev_p[0] = r

        for part in cls.NAME_BREAKUP_RE.split(token):
            if not part:
                continue
            prev = part[0]
            pos = 0
            for i in range(1, len(part)):
                this = part[i]
                if prev.islower() and this.isupper():
                    yield from ret(part[pos:i])
                    pos = i
                elif prev.isupper() and this.islower():
                    if 0 < i - 1 - pos <= 3:
                        yield from ret(part[pos:i - 1])
                        pos = i - 1
                    elif i - 1 > pos:
                        yield from ret(part[pos:i])
                        pos = i
                prev = this
            last = part[pos:]
            if last:
                yield from ret(last)


def repo2nbow(url_or_path, id2vec=None, df=None, linguist=None):
    if id2vec is None:
        id2vec = Id2Vec()
    if df is None:
        df = DocumentFrequencies()
    obj = Repo2nBOW(id2vec, df, linguist=linguist)
    nbow = obj.convert_repository(url_or_path)
    return nbow


def repo2nbow2stdout(args):
    id2vec = Id2Vec(args.id2vec or None)
    df = DocumentFrequencies(args.df or None)
    linguist = args.linguist or None
    nbow = repo2nbow(args.repository, id2vec=id2vec, df=df, linguist=linguist)
    nbl = [(weight, token) for token, weight in nbow.items()]
    nbl.sort(reverse=True)
    for w, t in nbl:
        print("%s\t%f\n" % (t, w))
