import warnings

import numpy as np
try:
    import keras
    from keras import backend as K
    from keras.layers import BatchNormalization, Concatenate, Conv1D, CuDNNGRU, CuDNNLSTM, Dense, \
        Embedding, GRU, Input, LSTM, TimeDistributed
    from keras.models import Model
    import tensorflow as tf
except ImportError as e:
    warnings.warn("Tensorflow or/and Keras are not installed, dependent functionality is "
                  "unavailable.")


LOSS = "binary_crossentropy"
METRICS = ["accuracy"]
DEFAULT_RNN_TYPE = "LSTM"


def register_metric(metric):
    assert isinstance(metric, str) or callable(metric)
    METRICS.append(metric)
    return metric


def prepare_input_emb(maxlen, n_uniq):
    """
    One-hot encoding of characters
    :param maxlen: maximum length of input sequence
    :param n_uniq: number of unique characters
    :return: tensor for input, one-hot character embeddings
    """
    char_seq = Input((maxlen,))
    emb = Embedding(input_dim=n_uniq + 1, output_dim=n_uniq + 1, input_length=maxlen,
                    mask_zero=False, weights=[np.eye(n_uniq + 1)], trainable=False)(char_seq)
    return char_seq, emb


def add_output_layer(input_layer):
    """
    Output layer has 1 sigmoid per character that should predict if there's a space before char
    :param input_layer: hidden layer before output layer
    :return: layer
    """
    norm_input = BatchNormalization()(input_layer)
    return TimeDistributed(Dense(1, activation="sigmoid"))(norm_input)


def add_rnn(X, units=128, rnn_layer=None, dev0="/gpu:0", dev1="/gpu:1"):
    """
    Add a RNN layer according to parameters.
    :param X: input layer
    :param units: number of neurons in layer
    :param rnn_layer: type of RNN layer
    :param dev0: device that will be used for forward pass of RNN and concatenation
    :param dev1: device that will be used for backward pass
    :return: layer
    """
    # select RNN layer
    rnn_layer_mapping = {"GRU": GRU, "LSTM": LSTM, "CuDNNLSTM": CuDNNLSTM, "CuDNNGRU": CuDNNGRU}

    if rnn_layer is None:
        rnn_layer = CuDNNLSTM
    elif isinstance(rnn_layer, str):
        rnn_layer = rnn_layer_mapping[rnn_layer]

    # add forward & backward RNN
    with tf.device(dev0):
        forward_gru = rnn_layer(units=units, return_sequences=True)(X)
    with tf.device(dev1):
        backward_gru = rnn_layer(units=units, return_sequences=True, go_backwards=True)(X)

    # concatenate
    with tf.device(dev1):
        bidi_gru = Concatenate(axis=-1)([forward_gru, backward_gru])
    return bidi_gru


def build_rnn(n_uniq, maxlen, units, stack, optimizer, rnn_layer, dev0, dev1):
    """
    Construct a RNN model according to given arguments.
    :param n_uniq: number of unique items/characters
    :param maxlen: maximum length of input sequence
    :param units: number of neurons in layer
    :param stack: number of RNN layers to stack
    :param optimizer: optimizer
    :param rnn_layer: type of RNN layer
    :param dev0: device to use in RNN
    :param dev1: device to use in RNN
    :return: compiled model
    """
    if rnn_layer is None:
        rnn_layer = DEFAULT_RNN_TYPE

    # prepare model
    with tf.device(dev0):
        char_seq, hid_layer = prepare_input_emb(maxlen, n_uniq)

        # stack BiDi-RNN
        for i in range(stack):
            hid_layer = add_rnn(hid_layer, units=units, rnn_layer=rnn_layer, dev0=dev0,
                                dev1=dev1)

        output = add_output_layer(hid_layer)

    # compile model
    model = Model(inputs=char_seq, outputs=output)
    model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
    return model


def add_conv(X, filters=[64, 32, 16, 8], kernel_sizes=[2, 4, 8, 16], output_n_filters=32):
    """
    Build a single convolutional layer.
    :param X: previous layer
    :param filters: number of filter for each kernel size
    :param kernel_sizes: list of kernel sizes
    :param output_n_filters: number of 1d output filters
    :return: layer
    """
    # normalization of input
    X = BatchNormalization()(X)

    # add convolutions
    convs = []

    for n_filters, kern_size in zip(filters, kernel_sizes):
        conv = Conv1D(filters=n_filters, kernel_size=kern_size, padding="same",
                      activation="relu")
        convs.append(conv(X))

    # concatenate all convolutions
    conc = Concatenate(axis=-1)(convs)
    conc = BatchNormalization()(conc)

    # dimensionality reduction
    conv = Conv1D(filters=output_n_filters, kernel_size=1, padding="same", activation="relu")
    return conv(conc)


def build_cnn(n_uniq, maxlen, filters, output_n_filters, stack, kernel_sizes, optimizer, device):
    """
    Construct a CNN model according to given arguments.
    :param n_uniq: number of unique items/characters
    :param maxlen: maximum length of input sequence
    :param filters: list with number of filters for each kernel size
    :param output_n_filters: number of 1-d filters that is applied after each CNN layer
    :param stack: number of CNN layers to stack
    :param kernel_sizes: list of kernel sizes
    :param optimizer: optimizer
    :param device: device to use
    :return: compiled model
    """
    # prepare model
    with tf.device(device):
        char_seq, hid_layer = prepare_input_emb(maxlen, n_uniq)

        # stack CNN
        for _ in range(stack):
            hid_layer = add_conv(hid_layer, filters=filters, kernel_sizes=kernel_sizes,
                                 output_n_filters=output_n_filters)

        output = add_output_layer(hid_layer)

    # compile model
    model = Model(inputs=char_seq, outputs=output)

    model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
    return model


@register_metric
def precision(y_true, y_pred):
    """
    Precision metric. Only computes a batch-wise average of recall.
    :param y_true: tensor
    :param y_pred: tensor
    :return: tensor
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


@register_metric
def recall(y_true, y_pred):
    """
    Recall metric. Only computes a batch-wise average of recall.
    :param y_true: tensor
    :param y_pred: tensor
    :return: tensor
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


@register_metric
def f1score(y_true, y_pred):
    """
    F1 score. Only computes a batch-wise average of recall.
    :param y_true: tensor
    :param y_pred: tensor
    :return: tensor
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec + K.epsilon())
