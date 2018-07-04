import string
from typing import Callable, List, Tuple, Union
import warnings

import numpy
import keras
from keras import backend as kbackend
from keras.layers import BatchNormalization, Concatenate, Conv1D, Dense, Embedding, \
    Input, TimeDistributed
from keras.models import Model
try:
    import tensorflow as tf
except ImportError:
    warnings.warn("Tensorflow is not installed, dependent functionality is unavailable.")


LOSS = "binary_crossentropy"
METRICS = ["accuracy"]
# Number of unique characters and dimension of the embedding layer
NUM_CHARS = len(string.ascii_lowercase)


def register_metric(metric: Union[str, Callable]) -> Union[str, Callable]:
    """
    Decorator function to register the metrics in the METRICS constant.

    :param metrics: name of the tensorflow metric or custom function metric.
    :return: the metric.
    """
    assert isinstance(metric, str) or callable(metric)
    METRICS.append(metric)
    return metric


def prepare_devices(devices: str) -> Tuple[str]:
    """
    Extract devices from arguments.

    :param devices: devices to use passed as one string argument.
    :return: splitted devices.
    """
    devices = devices.split(",")
    if len(devices) == 2:
        dev0, dev1 = ("/gpu:" + dev for dev in devices)
    elif len(devices) == 1:
        if int(devices[0]) != -1:
            dev0 = dev1 = "/gpu:" + devices[0]
        else:
            dev0 = dev1 = "/cpu:0"
    else:
        raise ValueError("Expected 1 or 2 devices but got %d from the devices argument %s" %
                         (len(devices), devices))
    return dev0, dev1


def prepare_input_emb(maxlen: int) -> Tuple[tf.Tensor]:
    """
    Builds character embeddings, a dense representation of characters to feed the RNN with.

    :param maxlen: maximum length of the input sequence.
    :param n_uniq: number of unique characters.
    :return: input and one-hot character embedding layer.
    """
    char_seq = Input((maxlen,))
    emb = Embedding(input_dim=NUM_CHARS + 1, output_dim=NUM_CHARS + 1, input_length=maxlen,
                    mask_zero=False, weights=[numpy.eye(NUM_CHARS + 1)], trainable=False)(char_seq)
    return char_seq, emb


def add_output_layer(hidden_layer: tf.Tensor) -> keras.layers.wrappers.TimeDistributed:
    """
    Applies a Dense layer to each of the timestamps of a hidden layer, independently.
    The output layer has 1 sigmoid per character which predicts if there is a space or not
    before the character.

    :param input_layer: hidden layer before the output layer.
    :return: output layer.
    """
    norm_input = BatchNormalization()(hidden_layer)
    return TimeDistributed(Dense(1, activation="sigmoid"))(norm_input)


def add_rnn(X: tf.Tensor, units: int, rnn_layer: str, dev0: str="/gpu:0",
            dev1: str="/gpu:1") -> tf.Tensor:
    """
    Adds a bidirectional RNN layer with the specified parameters.

    :param X: input layer.
    :param units: number of neurons in the output layer.
    :param rnn_layer: type of cell in the RNN.
    :param dev0: device that will be used as forward pass of RNN and concatenation.
    :param dev1: device that will be used as backward pass.
    :return: output bidirectional RNN layer.
    """
    # select the type of RNN layer
    rnn_layer = getattr(keras.layers, rnn_layer)

    # add the forward & backward RNN
    with tf.device(dev0):
        forward = rnn_layer(units=units, return_sequences=True)(X)
    with tf.device(dev1):
        backward = rnn_layer(units=units, return_sequences=True, go_backwards=True)(X)

    # concatenate
    with tf.device(dev1):
        bidi = Concatenate(axis=-1)([forward, backward])
    return bidi


def build_rnn(maxlen: int, units: int, stack: int, optimizer: str, dev0: str,
              dev1: str, rnn_layer: str) -> keras.engine.training.Model:
    """
    Builds a RNN model with the parameters specified as arguments.

    :param maxlen: maximum length of the input sequence.
    :param units: number of neurons or dimensionality of the output RNN.
    :param stack: number of RNN layers to stack.
    :param optimizer: algorithm to use as an optimizer for the RNN.
    :param rnn_layer: recurrent layer type to use.
    :param dev0: first device to use when running specific operations.
    :param dev1: second device to use when running specific operations.
    :return: compiled RNN model.
    """
    # prepare the model
    with tf.device(dev0):
        char_seq, hidden_layer = prepare_input_emb(maxlen)

        # stack the BiDi-RNN layers
        for _ in range(stack):
            hidden_layer = add_rnn(hidden_layer, units=units, rnn_layer=rnn_layer,
                                   dev0=dev0, dev1=dev1)
        output = add_output_layer(hidden_layer)

    # compile the model
    model = Model(inputs=char_seq, outputs=output)
    model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
    return model


def add_conv(X: tf.Tensor, filters: List[int], kernel_sizes: List[int],
             output_n_filters: int) -> tf.Tensor:
    """
    Builds a single convolutional layer.

    :param X: input layer.
    :param filters: number of output filters in the convolution.
    :param kernel_sizes: list of lengths of the 1D convolution window.
    :param output_n_filters: number of 1D output filters.
    :return: output layer.
    """
    # normalize the input
    X = BatchNormalization()(X)

    # add convolutions
    convs = []

    for n_filters, kernel_size in zip(filters, kernel_sizes):
        conv = Conv1D(filters=n_filters, kernel_size=kernel_size, padding="same",
                      activation="relu")
        convs.append(conv(X))

    # concatenate all convolutions
    conc = Concatenate(axis=-1)(convs)
    conc = BatchNormalization()(conc)

    # dimensionality reduction
    conv = Conv1D(filters=output_n_filters, kernel_size=1, padding="same", activation="relu")
    return conv(conc)


def build_cnn(maxlen: int, filters: List[int], output_n_filters: int, stack: int,
              kernel_sizes: List[int], optimizer: str, device: str) -> keras.engine.training.Model:
    """
    Builds a CNN model with the parameters specified as arguments.

    :param maxlen: maximum length of the input sequence.
    :param filters: number of output filters in the convolution.
    :param output_n_filters: number of 1d output filters.
    :param stack: number of CNN layers to stack.
    :param kernel_sizes: list of lengths of the 1D convolution window.
    :param optimizer: algorithm to use as an optimizer for the CNN.
    :param device: device to use when running specific operations.
    :return: compiled CNN model.
    """
    # prepare the model
    with tf.device(device):
        char_seq, hidden_layer = prepare_input_emb(maxlen)

        # stack the CNN layers
        for _ in range(stack):
            hidden_layer = add_conv(hidden_layer, filters=filters, kernel_sizes=kernel_sizes,
                                    output_n_filters=output_n_filters)
        output = add_output_layer(hidden_layer)

    # compile the model
    model = Model(inputs=char_seq, outputs=output)
    model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
    return model


@register_metric
def precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.

    :param y_true: tensor of true labels.
    :param y_pred: tensor of predicted labels.
    :return: a tensor batch-wise average of precision.
    """
    true_positives = kbackend.sum(kbackend.round(kbackend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = kbackend.sum(kbackend.round(kbackend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + kbackend.epsilon())
    return precision


@register_metric
def recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.

    :param y_true: tensor of true labels.
    :param y_pred: tensor of predicted labels.
    :return: a tensor batch-wise average of recall.
    """
    true_positives = kbackend.sum(kbackend.round(kbackend.clip(y_true * y_pred, 0, 1)))
    possible_positives = kbackend.sum(kbackend.round(kbackend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + kbackend.epsilon())
    return recall


@register_metric
def f1score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the F1 score,  the harmonic average of precision and recall.

    :param y_true: tensor of true labels.
    :param y_pred: tensor of predicted labels.
    :return: a tensor batch-wise average of F1 score.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec + kbackend.epsilon())
