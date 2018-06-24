import argparse
from datetime import datetime
import logging
import pickle
import random
import os
import warnings

import numpy
from keras import backend as kbackend
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, LearningRateScheduler
try:
    import tensorflow as tf
except ImportError:
    warnings.warn("Tensorflow is not installed, dependent functionality is unavailable.")
from typing import Tuple

from sourced.ml.algorithms.id_splitter.features import prepare_features


EPSILON = 10 ** -8
DEFAULT_THRESHOLD = 0.5  # threshold that is used to binarize predictions of the model


def set_random_seed(seed):
    """
    Fix random seed for reproducibility.
    :param seed: seed value
    """
    numpy.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


def to_binary(matrix, threshold: float, inplace: bool=True):
    """
    Helper function to binarize a matrix.
    :param mat: matrix or array
    :param threshold: if value >= threshold than it will be 1, else 0
    :param inplace: whether modify the matrix inplace or not
    :return: the binarized matrix
    """
    mask = matrix >= threshold
    if inplace:
        matrix_ = matrix
    else:
        matrix_ = matrix.copy()
    matrix_[mask] = 1
    matrix_[numpy.logical_not(mask)] = 0
    return matrix_


def precision_np(y_true: numpy.array, y_pred: numpy.array, epsilon: float=EPSILON) -> float:
    """
    Precision metric.
    :param y_true: ground truth labels - expect binary values
    :param y_pred: predicted labels - expect binary values
    :param epsilon: to avoid division by zero
    :return: precision
    """
    true_positives = np.sum(y_true * y_pred)
    predicted_positives = np.sum(y_pred)
    return true_positives / (predicted_positives + epsilon)


def recall_np(y_true: numpy.array, y_pred: numpy.array, epsilon: float=EPSILON) -> float:
    """
    Compute recall metric.
    :param y_true: matrix with ground truth labels - expect binary values
    :param y_pred: matrix with predicted labels - expect binary values
    :param epsilon: added to denominator to avoid division by zero
    :return: recall
    """
    true_positives = np.sum(y_true * y_pred)
    possible_positives = np.sum(y_true)
    return true_positives / (possible_positives + epsilon)


def report(model: keras.engine.training.Model, X, y, batch_size: int,
           threshold: float=DEFAULT_THRESHOLD, epsilon: float=EPSILON):
    """
    Prepare report for `model` on data `X` & `y`. It prints precision, recall, F1 score.

    :param model: model to apply
    :param X: features
    :param y: labels (expected binary labels)
    :param batch_size: batch size that will be used or prediction
    :param threshold: threshold to binarize predictions
    :param epsilon: added to denominator to avoid division by zero
    """
    log = logging.getLogger("report")

    # predict & skip the last dimension & binarize
    predictions = model.predict(X, batch_size=batch_size, verbose=1)[:, :, 0]
    predictions = to_binary(predictions, threshold)

    # report
    pr = precision_np(y[:, :, 0], predictions, epsilon=epsilon)
    rec = recall_np(y[:, :, 0], predictions, epsilon=epsilon)
    f1 = 2 * pr * rec / (pr + rec + epsilon)
    log.info("precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(pr, rec, f1))


def config_keras():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    kbackend.tensorflow_backend.set_session(tf.Session(config=config))


def prepare_train_generator(x, y, batch_size=500):
    assert x.shape[0] == y.shape[0], "Number of samples mismatch in x and y."

    def xy_generator():
        while True:
            n_batches = x.shape[0] // batch_size
            if n_batches * batch_size < x.shape[0]:
                n_batches += 1  # to yield last samples
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, x.shape[0])
                yield x[start:end], y[start:end]
    return xy_generator()


def prepare_schedule(lr, final_lr, n_epochs):
    delta = (lr - final_lr) / n_epochs

    def schedule(epoch):
        assert 0 <= epoch < n_epochs
        return lr - delta * epoch
    return schedule


def make_lr_scheduler(lr=0.001, final_lr=0.00001, n_epochs=10, verbose=1):
    """
    Prepare learning rate scheduler to change learning rate during the training.

    :param lr: initial learning rate
    :param final_lr: final learning rate
    :param n_epochs: number of epochs
    :param verbose: verbosity
    :return: LearningRateScheduler with linear schedul of learning rates
    """
    schedule = prepare_schedule(lr, final_lr, n_epochs)
    return LearningRateScheduler(schedule=schedule, verbose=verbose)


def prepare_devices(args: argparse.ArgumentParser):
    """
    Extract devices from arguments.

    :param args: arguments
    :return: splitted devices
    """
    devices = args.devices.split(",")
    if len(devices) == 2:
        dev0, dev1 = ("/gpu:" + dev for dev in devices)
    elif len(devices) == 1:
        if int(devices[0]) != -1:
            dev0 = dev1 = "/gpu:" + args.devices
        else:
            dev0 = dev1 = "/cpu:0"
    else:
        raise ValueError("Expected 1 or 2 devices but got {} from args.devices argument {}"
                         .format(len(devices), args.devices))

    return dev0, dev1


def prepare_callbacks(output_dir: str):
    """
    Prepare logging, tensorboard, model checkpoint callbacks and store their outputs in output_dir.
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
    return [tensorboard, csv_logger, model_saver]


def generator_parameters(batch_size, samples_per_epoch, n_samples, epochs):
    """
    Helper function to split huge dataset into smaller one to make reports more frequently.
    :param batch_size: batch size.
    :param samples_per_epoch: number of samples per mini-epoch or before each report.
    :param n_samples: total number of samples.
    :param epochs: number epochs over full dataset.
    :return: number of steps per epoch (should be used with generator) and number of sub-epochs
             where during sub-epoch only samples_per_epoch will be generated.
    """
    steps_per_epoch = samples_per_epoch // batch_size
    n_epochs = numpy.ceil(epochs * n_samples / samples_per_epoch)
    return steps_per_epoch, n_epochs


def train_id_splitter(args: argparse.ArgumentParser, model):
    log = logging.getLogger("train_id_splitter")
    config_keras()
    set_random_seed(args.seed)

    # prepare features
    x_train, x_test, y_train, y_test = prepare_features(csv_path=args.input,
                                                        use_header=args.include_csv_header,
                                                        identifiers_col=args.csv_token,
                                                        max_identifier_length=args.length,
                                                        split_identifiers_col=args.csv_token_split,
                                                        test_ratio=args.test_ratio,
                                                        padding=args.padding)

    # prepare train generator
    steps_per_epoch, n_epochs = generator_parameters(batch_size=args.batch_size,
                                                     samples_per_epoch=args.samples_before_report,
                                                     n_samples=x_train.shape[0],
                                                     epochs=args.epochs)
    train_gen = prepare_train_generator(x=x_train, y=y_train, batch_size=args.batch_size)

    # prepare test generator
    validation_steps, _ = generator_parameters(batch_size=args.val_batch_size,
                                               samples_per_epoch=x_test.shape[0],
                                               n_samples=x_test.shape[0],
                                               epochs=args.epochs)
    test_gen = prepare_train_generator(x=x_test, y=y_test, batch_size=args.val_batch_size)

    # initialize model
    model = model(args)
    log.info("Model summary:")
    model.summary(print_fn=log.info)

    # callbacks
    callbacks = prepare_callbacks(args.output)
    lr_scheduler = make_lr_scheduler(lr=args.lr, final_lr=args.final_lr, n_epochs=n_epochs)
    callbacks.append(lr_scheduler)

    # train
    history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                  validation_data=test_gen, validation_steps=validation_steps,
                                  callbacks=callbacks, epochs=n_epochs)

    # report quality on test dataset
    report(model, X=x_test, y=y_test, batch_size=args.val_batch_size)

    # save model & history
    with open(os.path.join(args.output, "model_history.pickle"), "wb") as f:
        pickle.dump(history.history, f)
    model.save(os.path.join(args.output, "last.model"))
    log.info("Completed!")
