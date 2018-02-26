import numpy


def extract_coocc_matrix(global_shape, word_indices, model):
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
