# source{d} ml [![Build Status](https://travis-ci.org/src-d/ml.svg)](https://travis-ci.org/src-d/ml) [![codecov](https://codecov.io/github/src-d/ml/coverage.svg?branch=develop)](https://codecov.io/gh/src-d/ml) [![PyPI](https://img.shields.io/pypi/v/sourcedml.svg)](https://pypi.python.org/pypi/sourcedml)

This project is the foundation for [MLoSC](https://github.com/src-d/awesome-machine-learning-on-source-code) research and development. It abstracts feature extraction and working with models, thus allowing to focus on the higher level tasks.

Currently, the following models are implemented:

* id2vec, source code identifier embeddings
* docfreq, source code identifier document frequencies (part of TF-IDF)
* nBOW, weighted bag of vectors, as in [src-d/wmd-relax](https://github.com/src-d/wmd-relax)
* topic modeling
* wmhlsh, locality sensitive hashing on top of weighted minhash

It is written in Python3 and has been tested on Linux and macOS. source{d} ml is tightly coupled with [source{d} engine](https://engine.sourced.tech) and delegates all the feature extraction to it.

Here is the list of projects which are built using sourced.ml:

* [vecino](https://github.com/src-d/vecino) - finding similar repositories
* [tmsc](https://github.com/src-d/tmsc) - topic modeling of repositories
* [role2vec](https://github.com/src-d/rol2vec) - AST node embedding and correction
* [snippet-ranger](https://github.com/src-d/snippet-ranger) - topic modeling of source code snippets

## Installation

```
pip3 install sourcedml
```

You need to have `libxml2` installed. E.g., on Ubuntu `apt install libxml2-dev`.

## Usage

This project exposes two interfaces: API and command line. The command line is

```
sourcedml --help
```

There is an example of using Python API [here](Doc/how_to_use_sourcedml.ipynb).

It exposes several tools to generate the models and setup the environment.

API is divided into two domains: models and training. The first is about using while the second
is about creating. Models: [Id2Vec](sourced/ml/id2vec.py),
[DocumentFrequencies](sourced/ml/df.py), [NBOW](sourced/ml/nbow.py), [Cooccurrences](sourced/ml/coocc.py).
Transformers (keras/sklearn style): [Repo2nBOWTransformer](sourced/ml/repo2/nbow.py#L72),
[Repo2CooccTransformer](sourced/ml/repo2/coocc.py#L101),
[PreprocessTransformer](sourced/ml/id_embedding.py#L22),
[SwivelTransformer](sourced/ml/id_embedding.py#L218) and
[PostprocessTransformer](sourced/ml/id_embedding.py#L241).

## Docker image

```
docker run -it --rm srcd/ml --help
```

If this first command fails with

```
Cannot connect to the Docker daemon. Is the docker daemon running on this host?
```

And you are sure that the daemon is running, then you need to add your user to `docker` group:
refer to the [documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

## Algorithms

#### Identifier embeddings

We build the source code identifier co-occurrence matrix for every repository.

1. Clone or read the repository from disk.
2. Classify files using [enry](https://github.com/src-d/enry).
3. Extract [UAST](https://doc.bblf.sh/uast/specification.html) from each supported file.
4. [Split and stem](sourced/ml/repo2/base.py#L160) all the identifiers in each tree.
5. [Traverse UAST](sourced/ml/repo2/coocc.py#L86), collapse all non-identifier paths and record all
identifiers on the same level as co-occurring. Besides, connect them with their immediate parents.
6. Write the individual co-occurrence matrices.
7. [Merge](sourced/ml/id_embedding.py#L50) co-occurrence matrices from all repositories. Write the
document frequencies model.
8. Train the embeddings using [Swivel](sourced/ml/swivel.py) running on Tensorflow. Interactively view
the intermediate results in Tensorboard using `--logs`.
9. Write the identifier embeddings model.
10. Publish generated models to the Google Cloud Storage.

1-6 is performed with `repo2coocc` tool / `Repo2CooccTransformer` class,
7 with `id2vec_preproc` / `id_embedding.PreprocessTransformer`, 8 with `id2vec_train` / `id_embedding.SwivelTransformer`,
9 with `id2vec_postproc` / `id_embedding.PostprocessTransformer` and 10 with `publish`.

#### Weighted Bag of Vectors

We represent every repository as a weighted bag-of-vectors, provided by we've got document
frequencies ("docfreq") and identifier embeddings ("id2vec").

1. Clone or read the repository from disk.
2. Classify files using [enry](https://github.com/src-d/enry).
3. Extract [UAST](https://doc.bblf.sh/uast/specification.html) from each supported file.
4. [Split and stem](sourced/ml/repo2/base.py#L160) all the identifiers in each tree.
5. Leave only those identifiers which are present in "docfreq" and "id2vec".
6. Set the weight of each such identifier as TF-IDF.
7. Set the value of each such identifier as the corresponding embedding vector.
8. Write the nBOW model.
9. Publish it to the Google Cloud Storage.

1-8 is performed with `repo2nbow` tool / `Repo2nBOWTransformer` class and 9 with `publish`.

#### Topic modeling

See [here](topic_modeling.md).

## Contributions
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

We use [PEP8](https://www.python.org/dev/peps/pep-0008/) with line length 99 and ". All the tests
must pass:

```
unittest discover /path/to/sourcedml
```

## License

Apache 2.0.
