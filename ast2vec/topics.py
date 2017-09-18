import sys
from typing import Union

from modelforge.model import split_strings, assemble_sparse_matrix, generate_meta, \
    write_model, merge_strings, disassemble_sparse_matrix, Model
from modelforge.models import register_model
import numpy
from scipy.sparse import csr_matrix

import ast2vec


@register_model
class Topics(Model):
    NAME = "topics"

    @property
    def tokens(self):
        return self._tokens

    @property
    def topics(self):
        """
        May be None if no topics are labeled.
        """
        return self._topics

    @property
    def matrix(self):
        """
        Rows: tokens
        Columns: topics
        """
        return self._matrix

    def construct(self, tokens: list, topics: Union[list, None], matrix):
        if len(tokens) != matrix.shape[1]:
            raise ValueError("Tokens and matrix do not match.")
        self._tokens = tokens
        self._topics = topics
        self._matrix = matrix
        return self

    def _load_tree(self, tree: dict) -> None:
        self.construct(split_strings(tree["tokens"]),
                       split_strings(tree["topics"]) if tree["topics"] else None,
                       assemble_sparse_matrix(tree["matrix"]))

    def dump(self) -> str:
        res = "%d topics, %d tokens\nFirst 10 tokens: %s\nTopics: " % (
            self.matrix.shape + (self.tokens[:10],))
        if self.topics is not None:
            res += "labeled, first 10: %s\n" % self.topics[:10]
        else:
            res += "unlabeled\n"
        nnz = self.matrix.getnnz()
        res += "non-zero elements: %d  (%f)" % (
            nnz, nnz / (self.matrix.shape[0] * self.matrix.shape[1]))
        return res

    def save(self, output, deps: Union[None, list]=None) -> None:
        if not deps:
            deps = self.meta["dependencies"]
        self._meta = generate_meta(self.NAME, ast2vec.__version__, *deps)
        write_model(self._meta,
                    {"tokens": merge_strings(self.tokens),
                     "topics": merge_strings(self.topics) if self.topics is not None else False,
                     "matrix": disassemble_sparse_matrix(self.matrix)},
                    output)

    def __len__(self):
        """
        Returns the number of topics.
        """
        return self.matrix.shape[0]

    def __getitem__(self, item):
        """
        Returns the keywords sorted by significance from topic index.
        """
        row = self.matrix[item]
        nnz = row.nonzero()[1]
        pairs = [(-row[0, i], i) for i in nnz]
        pairs.sort()
        return [(self.tokens[pair[1]], -pair[0]) for pair in pairs]

    def label_topics(self, labels):
        if len(labels) != len(self):
            raise ValueError("Sizes do not match: %d != %d" % (len(labels), len(self)))
        if not isinstance(labels[0], str):
            raise TypeError("Labels must be strings")
        self._topics = list(labels)


def bigartm2asdf_entry(args):
    """
    BigARTM "readable" model -> Topics -> Modelforge ASDF.
    """
    tokens = []
    data = []
    indices = []
    indptr = [0]
    if args.input != "-":
        fin = open(args.input)
    else:
        fin = sys.stdin
    try:
        # the first line is the header
        fin.readline()
        for line in fin:
            items = line.split(";")
            tokens.append(items[0])
            nnz = 0
            for i, v in enumerate(items[2:]):
                if v == "0":
                    continue
                nnz += 1
                data.append(float(v))
                indices.append(i)
            indptr.append(indptr[-1] + nnz)
    finally:
        if args.input != "-":
            fin.close()
    data = numpy.array(data, dtype=numpy.float32)
    indices = numpy.array(indices, dtype=numpy.int32)
    matrix = csr_matrix((data, indices, indptr), shape=(len(tokens), len(items) - 2)).T
    Topics().construct(tokens, None, matrix).save(args.output)
