import argparse
import string

from sourced.ml.cmd_entries import ArgumentDefaultsHelpFormatterNoNone
from sourced.ml.algorithms.id_splitter.nn_model import build_rnn_from_args, build_cnn_from_args, \
    DEFAULT_RNN_TYPE
from sourced.ml.algorithms.id_splitter.pipeline import train_id_splitter
from sourced.ml.utils.engine import pause


# Default parameter values from the paper https://arxiv.org/abs/1805.11651
MAXLEN = 40  # max length of sequence
PADDING = "post"  # add padding values after input
EPOCHS = 10
BATCH_SIZE = 500
VAL_BATCH_SIZE = 2000
START_LR = 0.001
FINAL_LR = 0.00001
DEFAULT_DEVICES = "0"
RANDOM_SEED = 1989
SAMPLES_BEFORE_REPORT = 5 * 10 ** 6
TEST_RATIO = 0.2  # fraction of dataset to use as test

# CSV default parameters
TOKEN_COL = 3
TOKEN_SPLIT_COL = 4

# RNN default parameters
RNN_TYPES = ("GRU", "LSTM", "CuDNNLSTM", "CuDNNGRU")

# CNN default parameters
FILTERS = "64,32,16,8"
KERNEL_SIZES = "2,4,8,16"
DIM_REDUCTION = 32


@pause
def train_id_splitter_bidirnn(args):
    return train_id_splitter(args, model=build_rnn_from_args)


@pause
def train_id_splitter_cnn(args):
    return train_is_splitter(args, model=build_cnn_from_args)


def add_train_id_splitter_args(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(help="Identifier splitter", dest="id-splitter")

    def add_parser(name, help_message):
        return subparsers.add_parser(
            name, help=help_message, formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    # common arguments for CNN/RNN models
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input data in CSV format:"
                        "num_files,num_occ,num_repos,token,token_split")
    parser.add_argument("-e", "--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs. The more the better"
                        "but the training time is proportional.")
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size. Higher values better utilize GPUs"
                        "but may harm the convergence.")
    parser.add_argument("-l", "--length", type=int, default=MAXLEN,
                        help="RNN sequence length.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the output directory where to store the results.")
    parser.add_argument("-t", "--test-ratio", type=float, default=TEST_RATIO,
                        help="Fraction of dataset to use as test.")
    parser.add_argument("-p", "--padding", default=PADDING, choices=("pre", "post"),
                        help="Pad either before or after each sequence.")
    # TODO: list available optimizers from keras and add their arguments
    parser.add_argument("--optimizer", default="Adam", choices=("RMSprop", "Adam"),
                        help="Optimizer to apply.")
    parser.add_argument("--lr", default=START_LR, type=float,
                        help="Initial learning rate.")
    parser.add_argument("--final-lr", default=FINAL_LR, type=float,
                        help="Final learning rate."
                        "The descent from the initial learning rate is done linearly.")
    parser.add_argument("--samples-before-report", type=int, default=SAMPLES_BEFORE_REPORT,
                        help="Number of samples between each validation report"
                             "and training updates.")
    parser.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE,
                        help="Batch size for validation."
                             "It can be increased to speed up the pipeline"
                             "but it proportionally increases the memory consumption.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed.")
    parser.add_argument("--devices", default=DEFAULT_DEVICES,
                        help="Device(s) to use. '-1' means CPU.")
    parser.add_argument("--csv-token", default=TOKEN_COL,
                        help="Column name with raw token.")
    parser.add_argument("--csv-token-split", default=TOKEN_SPLIT_COL,
                        help="Column name with splitted token.")
    header_help = "Treat the first line of the input CSV as a regular line."
    parser.add_argument("--include-csv-header", action="store_true", help=header_help)

    # RNN specific arguments
    rnn_parser = add_parser("rnn", "Train RNN model to split identifiers.")

    rnn_parser.set_defaults(handler=train_id_splitter_bidirnn)
    rnn_parser.add_argument("-t", "--type", default=DEFAULT_RNN_TYPE, choices=RNN_TYPES,
                            help="Recurrent layer type to use.")
    rnn_parser.add_argument("-n", "--neurons", default=256, type=int,
                            help="Number of neurons on each layer.")
    rnn_parser.add_argument("-s", "--stack", default=2, type=int,
                            help="Number of BiDi-RNN stacked on each other.")

    # CNN specific arguments
    cnn_parser = add_parser("cnn", "Train CNN model to split identifiers.")

    cnn_parser.set_defaults(handler=train_id_splitter_cnn)
    cnn_parser.add_argument("-f", "--filters", default=FILTERS,
                            help="Number of filters for each kernel size.")
    cnn_parser.add_argument("-s", "--stack", default=3, type=int,
                            help="Number of CNN layers stacked on each other.")
    cnn_parser.add_argument("-k", "--kernel-sizes", default=KERNEL_SIZES,
                            help="Sizes for sliding windows.")
    cnn_parser.add_argument("--dim-reduction", default=DIM_REDUCTION, type=int,
                            help="Number of 1-d kernels to reduce dimensionality after each "
                                 "layer.")
