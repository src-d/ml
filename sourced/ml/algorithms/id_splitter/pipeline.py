from datetime import datetime
import logging
import random
import os
from typing import Callable, Iterable, List, Tuple
import warnings

import numpy
import keras
from keras import backend as kbackend
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, LearningRateScheduler
try:
    import tensorflow as tf
except ImportError:
    warnings.warn("Tensorflow is not installed, dependent functionality is unavailable.")


# additional variable to avoid any division by zero when computing the precision and recall metrics
EPSILON = 10 ** -8
# threshold that is used to binarize predictions of the model
DEFAULT_THRESHOLD = 0.5


def set_random_seed(seed: int) -> None:
    """
    Fixes a random seed for reproducibility.

    :param seed: seed value.
    """
    numpy.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


def binarize(matrix: numpy.array, threshold: float, inplace: bool=True) -> numpy.array:
    """
    Helper function to binarize a matrix.

    :param matrix: matrix as a numpy.array.
    :param threshold: if value >= threshold then the value will be 1, else 0.
    :param inplace: whether to modify the matrix inplace or not.
    :return: the binarized matrix.
    """
    mask = matrix >= threshold
    if inplace:
        matrix_ = matrix
    else:
        matrix_ = matrix.copy()
    matrix_[mask] = 1
    matrix_[numpy.logical_not(mask)] = 0
    return matrix_


def str2ints(params: str) -> List[int]:
    """
    Convert a string with integer parameters to a list of integers.

    :param params: string that contains integer parameters separated by commas.
    :return: list of integers.
    """
    return list(map(int, params.split(",")))


def precision_np(y_true: numpy.array, y_pred: numpy.array, epsilon: float=EPSILON) -> float:
    """
    Computes the precision metric, a metric for multi-label classification of
    how many selected items are relevant.

    :param y_true: ground truth labels - expect binary values.
    :param y_pred: predicted labels - expect binary values.
    :param epsilon: added to the denominator to avoid any division by zero.
    :return: precision metric.
    """
    true_positives = numpy.sum(y_true * y_pred)
    predicted_positives = numpy.sum(y_pred)
    return true_positives / (predicted_positives + epsilon)


def recall_np(y_true: numpy.array, y_pred: numpy.array, epsilon: float=EPSILON) -> float:
    """
    Computes the recall metric, a metric for multi-label classification of
    how many relevant items are selected.

    :param y_true: matrix with ground truth labels - expect binary values.
    :param y_pred: matrix with predicted labels - expect binary values.
    :param epsilon: added to the denominator to avoid any division by zero.
    :return: recall metric.
    """
    true_positives = numpy.sum(y_true * y_pred)
    possible_positives = numpy.sum(y_true)
    return true_positives / (possible_positives + epsilon)


def report(model: keras.engine.training.Model, X: numpy.array, y: numpy.array, batch_size: int,
           threshold: float=DEFAULT_THRESHOLD, epsilon: float=EPSILON) -> None:
    """
    Prints a metric report of the `model` on the  data `X` & `y`.
    The metrics printed are precision, recall, F1 score.

    :param model: model considered.
    :param X: features.
    :param y: labels (expected binary labels).
    :param batch_size: batch size that will be used for prediction.
    :param threshold: threshold to binarize the predictions.
    :param epsilon: added to the denominator to avoid any division by zero.
    """
    log = logging.getLogger("report")

    # predict & skip the last dimension & binarize
    predictions = model.predict(X, batch_size=batch_size, verbose=1)[:, :, 0]
    predictions = binarize(predictions, threshold)

    # report
    pr = precision_np(y[:, :, 0], predictions, epsilon=epsilon)
    rec = recall_np(y[:, :, 0], predictions, epsilon=epsilon)
    f1 = 2 * pr * rec / (pr + rec + epsilon)
    log.info("precision: %.3f, recall: %.3f, f1: %.3f" % (pr, rec, f1))


def config_keras() -> None:
    """
    Initializes keras backend session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    kbackend.tensorflow_backend.set_session(tf.Session(config=config))


def build_train_generator(X: numpy.array, y: numpy.array,
                          batch_size: int=500) -> Iterable[Tuple[numpy.array]]:
    """
    Builds the generator that yields features and their labels.

    :param X: features.
    :param y: binary labels.
    :param batch_size: higher values better utilize GPUs.
    :return: generator of features and their labels.
    """
    assert X.shape[0] == y.shape[0], "Number of samples mismatch in X and y."

    def xy_generator():
        while True:
            n_batches = X.shape[0] // batch_size
            if n_batches * batch_size < X.shape[0]:
                n_batches += 1  # to yield last samples
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, X.shape[0])
                yield X[start:end], y[start:end]
    return xy_generator()


def build_schedule(lr: float, final_lr: float, n_epochs: int) -> Callable:
    """
    Builds the schedule of which the learning rate decreases.
    The schedule makes the learning rate decrease linearly.

    :param lr: initial learning rate.
    :param final_lr: final learning rate.
    :param n_epochs: number of training epochs.
    :return: the schedule of the learning rate.
    """
    delta = (lr - final_lr) / n_epochs

    def schedule(epoch: int) -> float:
        assert 0 <= epoch < n_epochs
        return lr - delta * epoch
    return schedule


def make_lr_scheduler(lr: float, final_lr: float, n_epochs: int,
                      verbose: int=1) -> keras.callbacks.LearningRateScheduler:
    """
    Prepares the scheduler to decrease the learning rate while training.

    :param lr: initial learning rate.
    :param final_lr: final learning rate.
    :param n_epochs: number of training epochs.
    :param verbose: level of verbosity.
    :return: LearningRateScheduler with linear schedule of the learning rate.
    """
    schedule = build_schedule(lr, final_lr, n_epochs)
    return LearningRateScheduler(schedule=schedule, verbose=verbose)


def prepare_callbacks(output_dir: str) -> Tuple[Callable]:
    """
    Prepares logging, tensorboard, model checkpoint callbacks and stores the outputs in output_dir.

    :param output_dir: path to the results.
    :return: list of callbacks.
    """
    time = datetime.now().strftime("%y%m%d-%H%M")
    log_dir = os.path.join(output_dir, "tensorboard" + time)
    logging.info("Tensorboard directory: %s" % log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, batch_size=1000, write_images=True,
                              write_graph=True)
    csv_path = os.path.join(output_dir, "csv_logger_" + time + ".txt")
    logging.info("CSV logs: %s" % csv_path)
    csv_logger = CSVLogger(csv_path)

    filepath = os.path.join(output_dir, "best_" + time + ".model")
    model_saver = ModelCheckpoint(filepath, monitor='val_recall', verbose=1, save_best_only=True,
                                  mode='max')
    return tensorboard, csv_logger, model_saver


def create_generator_params(batch_size: int, samples_per_epoch: int, n_samples: int,
                            epochs: int) -> Tuple[int]:
    """
    Helper function to split a huge dataset into smaller ones to enable more frequent reports.

    :param batch_size: batch size.
    :param samples_per_epoch: number of samples per mini-epoch or before each report.
    :param n_samples: total number of samples.
    :param epochs: number of epochs over the full dataset.
    :return: number of steps per epoch (should be used with the generator) and number of sub-epochs
             where during sub-epoch only samples_per_epoch will be generated.
    """
    steps_per_epoch = samples_per_epoch // batch_size
    n_epochs = numpy.ceil(epochs * n_samples / samples_per_epoch)
    return steps_per_epoch, n_epochs
