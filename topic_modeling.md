Topic Modeling of Repositories
------------------------------

Science: [paper](https://arxiv.org/abs/1704.00135)

Pipeline:

1. Collect the list of GitHub repositories to process.
2. Fetch repositories and save them as [`UASTModel`](ast2vec/uast.py) (a.k.a. `UAST` model).
3. Calculate document frequencies.
4. Produce BOW (bag-of-words) models from `UAST` models.
5. Join BOW models into the single BOW model.
6. Convert the BOW model to Vowpal Wabbit format.
7. Convert Vowpal Wabbit dataset to [BigARTM](https://github.com/bigartm/bigartm) batches.
8. Train the topic model using BigARTM.
9. Convert the result to [`TopicModel`](ast2vec/topic_model.py).

#### Collect the list of GitHub repositories to process

There are several options. You can use [GitHub API](https://developer.github.com/v3/)
or execute a query in
[BigQuery](https://cloud.google.com/bigquery/public-data/github). The easiest way is to download
the source{d}'s dataset with the whole world's open source projects (not released yet).

In the end, you should have a text file, say, `repos.txt` with a separate line per URL:

```
https://github.com/tensorflow/tensorflow
https://github.com/pytorch/pytorch
...
```

#### Fetch repositories and save them as `UAST` models

The first thing you need is to install [enry](https://github.com/src-d/enry),
source{d}'s source code classifer. The following command should produce `enry` executable
in the current directory. In the future, we suppose that we do not leave this directory.

```
ast2vec enry
```

Let's run the cloning pipeline:

```
ast2vec repos2uast -p 16 -t 4 --organize-files 2 -o uasts repos.txt
```

This will run 16 processes, each clones a repository, converts files to
Abstract Syntax Trees using [Babelfish](https://doc.bblf.sh/) in 4 threads and finally
writes the result to `uasts` directory.

The art of choosing `-p` and `-t` is hard to conceive. The general rule is to inspect the system
load and decide what is the current bottleneck:


| Cause                     | Symptoms                                   | Action                    |
|---------------------------|--------------------------------------------|---------------------------|
| cloning IO underload      | low CPU usage                              | increase `-p`             |
| network bandwidth limit   | CPU usage stays the same disregarding `-p` | increase `-t`             |
| Babelfish bandwidth limit | high CPU usage                             | decrease `-t`             |
| no free memory            | out-of-memory errors, swapping, lags       | decrease `-p` and/or `-t` |

In some cases Babelfish server responses take too much time and you get timeout errors.
Try to increase `--timeout` or if it does not help, decrease `-t` and even `-p`.

In the end, you will have `.asdf` files inside 2 levels of directories in `uasts`.

If resuming the pipeline, make sure to pass `--disable-overwrite` to not do the same work twice.

#### Calculate document frequencies

```
ast2vec uasts2df -p 4 uasts docfreq.asdf
```

We run 4 workers and save the result to `docfreq.asdf`.

#### Produce BOW (bag-of-words) models from `UAST` models

```
ast2vec uast2bow --df docfreq.asdf -v 100000 -p 4 uasts bows
```

Again, 4 workers. We set the number of distinct tokens to 100k here. The bigger the vocabulary size,
the better the model but the higher memory usage and bigger the bag-of-words models. It is sane to
increase `-v` up to 2-3 million.

The results will be in `bow` directory.

#### Join BOW models into the single BOW model

```
ast2vec join-bow -p 4 --bow bows joined_bow.asdf
```

4 workers merge the individual bags-of-words together into `joined_bow.asdf`.

#### Convert the BOW model to Vowpal Wabbit format

```
ast2vec bow2vw --bow joined_bow.asdf -o vw_dataset.txt
```

We transform the merged BOW model stored in ASDF binary format to simple text "Vowpal Wabbit" format.

The reason we use the intermediate format is that BigARTM's Python API is much slower at the direct
conversion.

#### Convert Vowpal Wabbit dataset to BigARTM batches

You will need a working `bigartm` command-line application. The following command should install
`bigartm` to the current working directory, provided by you have all the dependencies present in the
system.

```
ast2vec bigartm
```

The actual conversion:

```
./bigartm -c vw_dataset.txt -p 0 --save-batches artm_batches --save-dictionary artm_batches/artm.dict
```

#### Train the topic model using BigARTM

Stage 1 performs the main optimization:

```
./bigartm --use-batches artm_batches --use-dictionary artm_batches/artm.dict -t 256 -p 20 --threads 4 --rand-seed 777 --regularizer "1000 Decorrelation" --save-model stage1.bigartm

```

Stage 2 optimizes for sparsity:

```
./bigartm --use-batches artm_batches --use-dictionary artm_batches/artm.dict --load-model stage1.bigartm -p 10 --threads 4 --rand-seed 777 --regularizer "1000 Decorrelation" "0.5 SparsePhi" "0.5 SparseTheta" --save-model stage2.bigartm
```

We set the number of topics to 256, the number of workers to 4. `-p` sets the number of iterations (passes).
Choosing the stages and the regularizers is an art. Please refer to BigARTM papers.

#### Convert the result to `TopicModel`

First we convert the model to the text format:

```
./bigartm --use-batches artm_batches --use-dictionary artm_batches/artm.dict --load-model stage2.bigartm -p 0 --write-model-readable readable_stage2.txt
```

Second we convert the text format to the ASDF:

```
ast2vec bigartm2asdf readable_stage2.txt topic_model.asdf
```