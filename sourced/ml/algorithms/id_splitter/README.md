# Neural Identifier Splitter
Article [Splitting source code identifiers using Bidirectional LSTM Recurrent Neural Network](https://arxiv.org/abs/1805.11651).

### Agenda
* Data
* Training pipeline
* How to launch

### Data
You can download the dataset [here](https://drive.google.com/open?id=1wZR5zF1GL1fVcA1gZuAN_9rSLd5ssqKV). More information about the dataset is available [here](https://github.com/src-d/datasets/tree/master/Identifiers).
#### Data format
* format of file: `.csv.gz`.
* the `csv` structure:

|num_files|num_occ|num_repos|token|token_split|
|:--|:--|:--|:--|:--|
|1|2|1|quesesSet|queses set|
|...|...|...|...|...|

#### Data stats
* 49 millions of identifiers
* 1 GB

### Training pipeline
Training pipeline consists of several steps
* [prepare features](https://github.com/src-d/ml/blob/master/sourced/ml/algorithms/id_splitter/features.py#L44-#L118) - read data, extract features, train/test split
* [prepare generators for keras](https://github.com/src-d/ml/blob/master/sourced/ml/cmd/train_id_split.py#L34-#L48)
* [prepare model - RNN or CNN](https://github.com/src-d/ml/blob/master/sourced/ml/cmd/train_id_split.py#L53-#L76)
* [training](https://github.com/src-d/ml/blob/master/sourced/ml/cmd/train_id_split.py#L78-#L89)
* [quality report and save the model](https://github.com/src-d/ml/blob/master/sourced/ml/cmd/train_id_split.py#L91-#L96)

### How to launch
First of all you need to download data using link above.

Usage:
```console
usage: srcml train-id-split [-h] -i INPUT [-e EPOCHS] [-b BATCH_SIZE]
                            [-l LENGTH] -o OUTPUT [-t TEST_RATIO]
                            [-p {pre,post}] [--optimizer {RMSprop,Adam}]
                            [--lr LR] [--final-lr FINAL_LR]
                            [--samples-before-report SAMPLES_BEFORE_REPORT]
                            [--val-batch-size VAL_BATCH_SIZE] [--seed SEED]
                            [--devices DEVICES]
                            [--csv-identifier CSV_IDENTIFIER]
                            [--csv-identifier-split CSV_IDENTIFIER_SPLIT]
                            [--include-csv-header] --model {RNN,CNN}
                            [-s STACK]
                            [--type-cell {GRU,LSTM,CuDNNLSTM,CuDNNGRU}]
                            [-n NEURONS] [-f FILTERS] [-k KERNEL_SIZES]
                            [--dim-reduction DIM_REDUCTION]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input data in CSV
                        format:num_files,num_occ,num_repos,token,token_split
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs. The more the betterbut the
                        training time is proportional. (default: 10)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size. Higher values better utilize GPUsbut may
                        harm the convergence. (default: 500)
  -l LENGTH, --length LENGTH
                        RNN sequence length. (default: 40)
  -o OUTPUT, --output OUTPUT
                        Path to store the trained model.
  -t TEST_RATIO, --test-ratio TEST_RATIO
                        Fraction of the dataset to use for evaluation.
                        (default: 0.2)
  -p {pre,post}, --padding {pre,post}
                        Whether to pad before or after each sequence.
                        (default: post)
  --optimizer {RMSprop,Adam}
                        Algorithm to use as an optimizer for the neural net.
                        (default: Adam)
  --lr LR               Initial learning rate. (default: 0.001)
  --final-lr FINAL_LR   Final learning rate. The decrease from the initial
                        learning rate is done linearly. (default: 1e-05)
  --samples-before-report SAMPLES_BEFORE_REPORT
                        Number of samples between each validation reportand
                        training updates. (default: 5000000)
  --val-batch-size VAL_BATCH_SIZE
                        Batch size for validation.It can be increased to speed
                        up the pipeline butit proportionally increases the
                        memory consumption. (default: 2000)
  --seed SEED           Random seed. (default: 1989)
  --devices DEVICES     Device(s) to use. '-1' means CPU. (default: 0)
  --csv-identifier CSV_IDENTIFIER
                        Column name in the CSV file for the raw identifier.
                        (default: 3)
  --csv-identifier-split CSV_IDENTIFIER_SPLIT
                        Column name in the CSV file for the splitidentifier.
                        (default: 4)
  --include-csv-header  Treat the first line of the input CSV as a
                        regularline. (default: False)
  --model {RNN,CNN}     Neural Network model to use to learn the
                        identifiersplitting task.
  -s STACK, --stack STACK
                        Number of layers stacked on each other. (default: 2)
  --type-cell {GRU,LSTM,CuDNNLSTM,CuDNNGRU}
                        Recurrent layer type to use. (default: LSTM)
  -n NEURONS, --neurons NEURONS
                        Number of neurons on each layer. (default: 256)
  -f FILTERS, --filters FILTERS
                        Number of filters for each kernel size. (default:
                        64,32,16,8)
  -k KERNEL_SIZES, --kernel-sizes KERNEL_SIZES
                        Sizes for sliding windows. (default: 2,4,8,16)
  --dim-reduction DIM_REDUCTION
                        Number of 1-d kernels to reduce dimensionalityafter
                        each layer. (default: 32)
```


Examples of commands:
1) Train RNN with LSTM cells
```console
srcml train-id-split --model RNN --input /path/to/input.csv.gz --output /path/to/output
```
2) Train RNN with CuDNNLSTM cells
```console
srcml train-id-split --model RNN --input /path/to/input.csv.gz --output /path/to/output  \
--type-cell CuDNNLSTM
```
3) Train CNN
```console
srcml train-id-split --model CNN --input /path/to/input.csv.gz --output /path/to/output
```
