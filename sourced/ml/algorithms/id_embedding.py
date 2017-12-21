from argparse import Namespace

import numpy

import sourced.ml.algorithms.swivel as swivel
from sourced.ml.cmd_entries import postprocess_id2vec, preprocess_id2vec
from sourced.ml.cmd_entries.run_swivel import run_swivel


class Transformer:
    pass


class PreprocessTransformer(Transformer):
    vocabulary_size = 1 << 17
    shard_size = 4096

    def __init__(self, vocabulary_size=None, shard_size=None):
        super().__init__()
        if vocabulary_size is not None:
            self.vocabulary_size = vocabulary_size
        if shard_size is not None:
            self.shard_size = shard_size

    def transform(self, X, output, df=None, vocabulary_size=None,
                  shard_size=None):
        if vocabulary_size is not None:
            self.vocabulary_size = vocabulary_size
        if shard_size is not None:
            self.shard_size = shard_size

        if isinstance(X, str):
            X = [X]

        args = Namespace(vocabulary_size=self.vocabulary_size,
                         input=X, df=df, shard_size=self.shard_size,
                         output=output)
        preprocess_id2vec(args)

    def _get_log_name(self):
        return "id_preprocess"


def _extract_coocc_matrix(global_shape, word_indices, model):
    # Stage 1 - extract the tokens, map them to the global vocabulary
    indices = []
    mapped_indices = []
    for i, w in enumerate(model.tokens):
        gi = word_indices.get(w)
        if gi is not None:
            indices.append(i)
            mapped_indices.append(gi)
    indices = numpy.array(indices)
    mapped_indices = numpy.array(mapped_indices)
    # Stage 2 - sort the matched tokens by the index in the vocabulary
    order = numpy.argsort(mapped_indices)
    indices = indices[order]
    mapped_indices = mapped_indices[order]
    # Stage 3 - produce the csr_matrix with the matched tokens **only**
    matrix = model.matrix.tocsr()[indices][:, indices]
    # Stage 4 - convert this matrix to the global (ccmatrix) coordinates
    csr_indices = matrix.indices
    for i, v in enumerate(csr_indices):
        # Here we use the fact that indices and mapped_indices are in the same order
        csr_indices[i] = mapped_indices[v]
    csr_indptr = matrix.indptr
    new_indptr = [0]
    for i, v in enumerate(mapped_indices):
        prev_ptr = csr_indptr[i]
        ptr = csr_indptr[i + 1]

        # Handle missing rows
        prev = (mapped_indices[i - 1] + 1) if i > 0 else 0
        for z in range(prev, v):
            new_indptr.append(prev_ptr)

        new_indptr.append(ptr)
    for z in range(mapped_indices[-1] + 1, global_shape[0]):
        new_indptr.append(csr_indptr[-1])
    matrix.indptr = numpy.array(new_indptr)
    matrix._shape = global_shape
    return matrix


class SwivelTransformer(Transformer):
    def transform(self, **kwargs):
        flags = type(swivel.FLAGS)()
        flags.__dict__ = swivel.FLAGS.__dict__.copy()

        for key, val in kwargs.items():
            if val is not None:
                setattr(flags, key, val)

        run_swivel(flags)

    def _get_log_name(self):
        return "id_swivel"


class PostprocessTransformer(Transformer):
    def transform(self, swivel_output_directory, result):
        """
        Merges row and column embeddings produced by Swivel and writes the
        Id2Vec model.

        :param swivel_output_directory: directory that contains files after swivel training. The \
                                        files are read from this directory and the model is \
                                        written to the 'result'.
        :param result: file to store results
        :return: None
        """
        args = Namespace(swivel_output_directory=swivel_output_directory,
                         result=result)
        postprocess_id2vec(args)

    def _get_log_name(self):
        return "id_postprocess"
