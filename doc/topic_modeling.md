Topic Modeling of Repositories
------------------------------

Science: [paper](https://arxiv.org/abs/1704.00135)

Pipeline:

1. Collect GitHub repositories you wish to process. E.g. from [Public Git Archive](https://github.com/src-d/datasets/tree/master/PublicGitArchive).
2. Produce BOW (bag-of-words) model from the identifiers inside those repositories.
3. Convert the BOW model to Vowpal Wabbit format.
4. Convert Vowpal Wabbit dataset to [BigARTM](https://github.com/bigartm/bigartm) batches.
5. Train the topic model using BigARTM.
6. Convert the result to [`Topics`](sourced/ml/models/topics.py) ASDF.

#### Generating BOW

Ensure that the [Babelfish server is running](https://doc.bblf.sh/user/getting-started.html).

```
srcml repos2bow -f id -x repo -l Java Python Ruby --min-docfreq 5 --persist DISK_ONLY --docfreq docfreq.asdf --bow bow.asdf
```

Change "Java Python Ruby" to any list of languages you want to process and are [annotated by Babelfish](https://doc.bblf.sh/languages.html).
It is possible to run it on a Spark cluster, in that case specify `--spark`.

#### Convert the BOW model to Vowpal Wabbit format

```
srcml bow2vw --bow bow.asdf -o vw_dataset.txt
```

We transform the merged BOW model stored in ASDF binary format to simple text "Vowpal Wabbit" format.

The reason we use the intermediate format is that BigARTM's Python API is much slower at the direct
conversion.

#### Convert Vowpal Wabbit dataset to BigARTM batches

You will need a working `bigartm` command-line application. The following command should install
`bigartm` to the current working directory, provided by you have all the dependencies present in the
system.

```
srcml bigartm
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
srcml bigartm2asdf readable_stage2.txt topic_model.asdf
```