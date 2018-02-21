import logging
from typing import Iterable, List, Dict

from scipy import sparse

from modelforge import Model, split_strings, assemble_sparse_matrix, \
    merge_strings, disassemble_sparse_matrix, register_model
from modelforge.progress_bar import progress_bar
from sourced.ml.models.df import DocumentFrequencies


@register_model
class BOW(Model):
    """
    Weighted bag of words model. Every word is correspond to an index and its matrix column.
    Bag is a word set from repository, file or anything else.
    Word is source code identifier or its part.
    This model depends on :class:`sourced.ml.models.DocumentFrequencies`.
    """
    NAME = "bow"

    def construct(self, documents: List[str], tokens: List[str], matrix: sparse.spmatrix):
        if matrix.shape[0] != len(documents):
            raise ValueError("matrix shape mismatch, documents %d != %d" % (
                matrix.shape[0], len(documents)))
        if matrix.shape[1] != len(tokens):
            raise ValueError("matrix shape mismatch, tokens %d != %d" % (
                matrix.shape[1], len(tokens)))
        self._documents = documents
        self._matrix = matrix
        self._tokens = tokens
        return self

    def dump(self):
        return "Shape: %s\n" \
               "First 10 documents: %s\n" \
               "First 10 tokens: %s" % \
               (self._matrix.shape, self._documents[:10], self.tokens[:10])

    @property
    def matrix(self) -> sparse.spmatrix:
        """
        Returns the bags as a sparse matrix. Rows are documents and columns are tokens weight.
        """
        return self._matrix

    @property
    def documents(self):
        """
        The list of documents in the model.
        """
        return self._documents

    @property
    def tokens(self):
        """
        The list of tokens in the model.
        """
        return self._tokens

    def __getitem__(self, item: int):
        """
        Returns document name, word indices and weights for the given document index.

        :param item: Document index.
        :return: (name, :class:`numpy.ndarray` with word indices, \
                  :class:`numpy.ndarray` with weights)
        """
        data = self._matrix[item]
        return self._documents[item], data.indices, data.data

    def __iter__(self):
        """
        Returns an iterator over the document indices.
        """
        return iter(range(len(self)))

    def __len__(self):
        """
        Returns the number of documents.
        """
        return len(self._documents)

    def save(self, output: str, deps: Iterable=tuple()):
        if not deps:
            try:
                deps = [self.get_dep(DocumentFrequencies.NAME)]
            except KeyError:
                raise ValueError(
                    "You must specify DocumentFrequencies dependency to save BOW.") from None
        super().save(output, deps)

    def convert_bow_to_vw(self, output: str):
        log = logging.getLogger("bow2vw")
        log.info("Writing %s", output)
        with open(output, "w") as fout:
            for index in progress_bar(self, log, expected_size=len(self)):
                record = self[index]
                fout.write(record[0].replace(":", "").replace(" ", "_") + " ")
                pairs = []
                for t, v in zip(*record[1:]):
                    try:
                        word = self.tokens[t]
                    except (KeyError, IndexError):
                        log.warning("%d not found in the vocabulary", t)
                        continue
                    pairs.append("%s:%s" % (word, v))
                fout.write(" ".join(pairs))
                fout.write("\n")

    def documents_index(self) -> Dict[str, int]:
        return {r: i for i, r in enumerate(self._documents)}

    def _generate_tree(self):
        return {"documents": merge_strings(self._documents),
                "matrix": disassemble_sparse_matrix(self._matrix),
                "tokens": merge_strings(self.tokens)}

    def _load_tree_kwargs(self, tree: dict):
        return dict(documents=split_strings(tree["documents"]),
                    matrix=assemble_sparse_matrix(tree["matrix"]),
                    tokens=split_strings(tree["tokens"]))

    def _load_tree(self, tree: dict):
        self.construct(**self._load_tree_kwargs(tree))
