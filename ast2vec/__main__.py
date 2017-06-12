import sys

import numpy

from ast2vec.swivel import parse_swivel_embeddings


def swivel_embeddings_to_npz(input, output):
    embeddings, tokens = parse_swivel_embeddings(input)
    numpy.savez_compressed(output, embeddings=embeddings, tokens=tokens)


def main():
    pass

if __name__ == "__main__":
    sys.exit(main())
