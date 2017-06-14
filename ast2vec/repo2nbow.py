import json
import logging
import os
import re
import shutil
import subprocess
import tempfile

import Stemmer


class Repo2nBOW:
    NAME_BREAKUP_RE = re.compile(r"[^a-zA-Z]+")
    STEM_THRESHOLD = 6

    def __init__(self, id2vec, docfreq, tempdir=None, linguist=None,
                 log_level=logging.INFO):
        self._log = logging.getLogger("repo2nbow")
        self._log.setLevel(log_level)
        self._id2vec = id2vec
        self._docfreq = docfreq
        self._stemmer = Stemmer.Stemmer("english")
        self._stemmer.maxCacheSize = 0
        self._stem_threshold = 6
        self._tempdir = tempdir
        self._linguist = "enry" if linguist is None else linguist

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
            classified = self._classify_files(target_dir)
            // TODO: babelfish
        finally:
            if temp:
                shutil.rmtree(target_dir)

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
