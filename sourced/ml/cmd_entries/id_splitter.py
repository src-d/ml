import argparse

from sourced.ml.cmd_entries import ArgumentDefaultsHelpFormatterNoNone
from sourced.ml.algorithms.id_splitter.nn_model import build_rnn_from_args, build_cnn_from_args
from sourced.ml.algorithms.id_splitter.pipeline import train_id_splitter
from sourced.ml.utils.engine import pause


# Default parameter values from the paper https://arxiv.org/abs/1805.11651
DEFAULT_MAX_IDENTIFIER_LEN = 40  # default max length of the sequences.
PADDING = "post"  # add padding values after the input.
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 500
DEFAULT_VAL_BATCH_SIZE = 2000
DEFAULT_START_LR = 0.001
DEFAULT_FINAL_LR = 0.00001
DEFAULT_DEVICES = "0"
DEFAULT_RANDOM_SEED = 1989
DEFAULT_SAMPLES_BEFORE_REPORT = 5 * 10 ** 6
DEFAULT_TEST_RATIO = 0.2  # fraction of dataset to use as test

# In the CSV file, columns 0,1,2 contain statistics about the identifier.
CSV_IDENTIFIER_COL = 3  # Column 3 contains the input identifier e.g. "FooBar".
CSV_SPLIT_IDENTIFIER_COL = 4  # Column 4 contains the identifier lowercase and spitted "foo bar".

# RNN default parameters
RNN_TYPES = ("GRU", "LSTM", "CuDNNLSTM", "CuDNNGRU")
DEFAULT_RNN_TYPE = "LSTM"
DEFAULT_RNN_STACK = 2
DEFAULT_NEURONS = 256

# CNN default parameters
DEFAULT_FILTERS = "64,32,16,8"
DEFAULT_KERNEL_SIZES = "2,4,8,16"
DEFAULT_DIM_REDUCTION = 32
DEFAULT_SHUFFLE_VALUE = True
DEFAULT_CNN_STACK = 3


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
    parser.add_argument("-e", "--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of training epochs. The more the better"
                        "but the training time is proportional.")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size. Higher values better utilize GPUs"
                        "but may harm the convergence.")
    parser.add_argument("-l", "--length", type=int, default=DEFAULT_MAX_IDENTIFIER_LEN,
                        help="RNN sequence length.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the output directory where to store the results.")
    parser.add_argument("-t", "--test-ratio", type=float, default=DEFAULT_TEST_RATIO,
                        help="Fraction of dataset to use as test.")
    parser.add_argument("-p", "--padding", default=PADDING, choices=("pre", "post"),
                        help="Pad either before or after each sequence.")
    # TODO: list available optimizers from keras and add their arguments
    parser.add_argument("--optimizer", default="Adam", choices=("RMSprop", "Adam"),
                        help="Optimizer to apply.")
    parser.add_argument("--lr", default=DEFAULT_START_LR, type=float,
                        help="Initial learning rate.")
    parser.add_argument("--final-lr", default=DEFAULT_FINAL_LR, type=float,
                        help="Final learning rate."
                        "The descent from the initial learning rate is done linearly.")
    parser.add_argument("--samples-before-report", type=int, default=DEFAULT_SAMPLES_BEFORE_REPORT,
                        help="Number of samples between each validation report"
                             "and training updates.")
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_VAL_BATCH_SIZE,
                        help="Batch size for validation."
                             "It can be increased to speed up the pipeline"
                             "but it proportionally increases the memory consumption.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random seed.")
    parser.add_argument("--devices", default=DEFAULT_DEVICES,
                        help="Device(s) to use. '-1' means CPU.")
    parser.add_argument("--csv-identifier", default=CSV_IDENTIFIER_COL,
                        help="Column name in the CSV file for the raw identifier.")
    parser.add_argument("--csv-identifier-split", default=CSV_SPLIT_IDENTIFIER_COL,
                        help="Column name in the CSV file for the splitted identifier.")
    header_help = "Treat the first line of the input CSV as a regular line."
    parser.add_argument("--include-csv-header", action="store_true", help=header_help)

    # RNN specific arguments
    rnn_parser = add_parser("rnn", "Train RNN model to split identifiers.")

    rnn_parser.set_defaults(handler=train_id_splitter_bidirnn)
    rnn_parser.add_argument("-t", "--type", default=DEFAULT_RNN_TYPE, choices=RNN_TYPES,
                            help="Recurrent layer type to use.")
    rnn_parser.add_argument("-n", "--neurons", default=DEFAULT_NEURONS, type=int,
                            help="Number of neurons on each layer.")
    rnn_parser.add_argument("-s", "--stack", default=DEFAULT_RNN_STACK, type=int,
                            help="Number of BiDi-RNN stacked on each other.")

    # CNN specific arguments
    cnn_parser = add_parser("cnn", "Train CNN model to split identifiers.")

    cnn_parser.set_defaults(handler=train_id_splitter_cnn)
    cnn_parser.add_argument("-f", "--filters", default=DEFAULT_FILTERS,
                            help="Number of filters for each kernel size.")
    cnn_parser.add_argument("-s", "--stack", default=DEFAULT_CNN_STACK, type=int,
                            help="Number of CNN layers stacked on each other.")
    cnn_parser.add_argument("-k", "--kernel-sizes", default=DEFAULT_KERNEL_SIZES,
                            help="Sizes for sliding windows.")
    cnn_parser.add_argument("--dim-reduction", default=DEFAULT_DIM_REDUCTION, type=int,
                            help="Number of 1-d kernels to reduce dimensionality after each "
                                 "layer.")
