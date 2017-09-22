import logging

from modelforge import generate_meta
from modelforge.model import Model, split_strings, write_model, merge_strings
from modelforge.models import register_model
import numpy

import ast2vec
import ast2vec.projector as projector


@register_model
class Id2Vec(Model):
    """
    id2vec model - source code identifier embeddings.
    """
    NAME = "id2vec"

    def construct(self, embeddings, tokens):
        self._embeddings = embeddings
        self._tokens = tokens
        self._log.info("Building the token index...")
        self._token2index = {w: i for i, w in enumerate(self._tokens)}
        return self

    def _load_tree(self, tree):
        self.construct(embeddings=tree["embeddings"].copy(),
                       tokens=split_strings(tree["tokens"]))

    def dump(self):
        return """Shape: %s
First 10 words: %s""" % (
            self.embeddings.shape, self.tokens[:10])

    @property
    def embeddings(self):
        """
        :class:`numpy.ndarray` with the embeddings of shape
        (N tokens x embedding dims).
        """
        return self._embeddings

    @property
    def tokens(self):
        """
        List with the processed source code identifiers.
        """
        return self._tokens

    def items(self):
        """
        Returns the tuples belonging to token -> index mapping.
        """
        return self._token2index.items()

    def __getitem__(self, item):
        """
        Returns the index of the specified processed source code identifier.
        """
        return self._token2index[item]

    def __len__(self):
        """
        Returns the number of tokens in the model.
        """
        return len(self._tokens)

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, ast2vec.__version__, *deps)
        write_model(self._meta,
                    {"embeddings": self.embeddings, "tokens": merge_strings(self.tokens)},
                    output)


def projector_entry(args):
    MAX_TOKENS = 10000  # hardcoded in Tensorflow Projector

    log = logging.getLogger("id2vec_projector")
    id2vec = Id2Vec(log_level=args.log_level).load(source=args.input)
    if args.df:
        from ast2vec.df import DocumentFrequencies
        df = DocumentFrequencies(log_level=args.log_level).load(source=args.df)
    else:
        df = None
    if len(id2vec) < MAX_TOKENS:
        tokens = numpy.arange(len(id2vec), dtype=int)
        if df is not None:
            freqs = [df.get(id2vec.tokens[i], 0) for i in tokens]
        else:
            freqs = None
    else:
        if df is not None:
            log.info("Filtering tokens through docfreq")
            items = []
            for token, idx in id2vec.items():
                try:
                    items.append((df[token], idx))
                except KeyError:
                    continue
            log.info("Sorting")
            items.sort(reverse=True)
            tokens = [i[1] for i in items[:MAX_TOKENS]]
            freqs = [i[0] for i in items[:MAX_TOKENS]]
        else:
            log.warning("You have not specified --df => picking random %d tokens", MAX_TOKENS)
            numpy.random.seed(777)
            tokens = numpy.random.choice(
                numpy.arange(len(id2vec), dtype=int), MAX_TOKENS, replace=False)
            freqs = None
    log.info("Gathering the embeddings")
    embeddings = numpy.vstack([id2vec.embeddings[i] for i in tokens])
    tokens = [id2vec.tokens[i] for i in tokens]
    labels = ["subtoken"]
    if freqs is not None:
        labels.append("docfreq")
        tokens = list(zip(tokens, (str(i) for i in freqs)))
    projector.present_embeddings(args.output, not args.no_browser, labels, tokens, embeddings)
    if not args.no_browser:
        projector.wait()
