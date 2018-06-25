import argparse
import tempfile
import unittest
import warnings

import numpy
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.backend.tensorflow_backend import get_session

from sourced.ml.cmd.args import ArgumentDefaultsHelpFormatterNoNone
from sourced.ml.cmd_entries.id_splitter import add_train_id_splitter_args
from sourced.ml.algorithms.id_splitter import build_schedule, \
    prepare_callbacks, build_train_generator, to_binary, \
    generator_parameters, config_keras, train_id_splitter
from sourced.ml.algorithms.id_splitter import register_metric, METRICS, \
    build_cnn_from_args, build_rnn_from_args, prepare_devices
from sourced.ml.tests.models import IDENTIFIERS


class Fake:
    pass


class IdSplitterPipelineTest(unittest.TestCase):
    def test_to_binary(self):
        thresholds = [0, 0.09, 0.19, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89, 0.99]
        n_pos = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        for th, n_p in zip(thresholds, n_pos):
            vals = numpy.arange(10) / 10
            res = to_binary(vals, th)
            self.assertEqual(sum(to_binary(vals, th)), n_p)
            if th in (0, 0.99):
                self.assertTrue(numpy.unique(res).shape[0] == 1)
            else:
                self.assertTrue(numpy.unique(res).shape[0] == 2)

        vals = numpy.arange(10) / 10
        old_vals = vals.copy()
        for th, n_p in zip(thresholds, n_pos):
            res = to_binary(vals, th, inplace=False)
            self.assertEqual(sum(res), n_p)
            self.assertTrue(numpy.array_equal(old_vals, vals))
            if th in (0, 0.99):
                self.assertTrue(numpy.unique(res).shape[0] == 1)
            else:
                self.assertTrue(numpy.unique(res).shape[0] == 2)

    def test_prepare_devices(self):
        correct_args = ["1", "0,1", "-1"]
        resulted_dev = [("/gpu:1", "/gpu:1"), ("/gpu:0", "/gpu:1"), ("/cpu:0", "/cpu:0")]
        for res, arg in zip(resulted_dev, correct_args):
            self.assertEqual(res, prepare_devices(arg))

        bad_args = ["", "1,2,3"]
        for arg in bad_args:
            with self.assertRaises(ValueError):
                prepare_devices(arg)

    def test_build_schedule(self):
        start_lr = 10
        end_lr = 1
        n_epochs = 9

        lr_schedule = build_schedule(lr=start_lr, final_lr=end_lr, n_epochs=n_epochs)

        for i in range(n_epochs):
            self.assertEqual(start_lr - i, lr_schedule(epoch=i))

        with self.assertRaises(AssertionError):
            lr_schedule(-1)
        with self.assertRaises(AssertionError):
            lr_schedule(n_epochs + 1)

    def test_build_train_generator(self):
        batch_size = 3
        # mismatch number of samples
        bad_x = numpy.zeros(3)
        bad_y = numpy.zeros(4)
        with self.assertRaises(AssertionError):
            build_train_generator(bad_x, bad_y, batch_size=batch_size)

        # check generator with correct inputs
        x = numpy.zeros(5)
        gen = build_train_generator(x, x, batch_size=batch_size)
        expected_n_samples = [3, 2]
        for n_samples in expected_n_samples:
            x_gen, y_gen = next(gen)
            self.assertEqual(x_gen.shape, y_gen.shape)
            self.assertEqual(n_samples, x_gen.shape[0])

    def test_train_parameters(self):
        batch_size = 500
        samples_per_epoch = 10 ** 6
        n_samples = 40 * 10 ** 6
        epochs = 10

        steps_per_epoch_ = samples_per_epoch // batch_size
        n_epochs_ = numpy.ceil(epochs * n_samples / samples_per_epoch)

        steps_per_epoch, n_epochs = generator_parameters(batch_size, samples_per_epoch, n_samples,
                                                         epochs)
        self.assertEqual(steps_per_epoch, steps_per_epoch_)
        self.assertEqual(n_epochs, n_epochs_)

    def test_config_keras(self):
        config_keras()
        sess = get_session()
        self.assertTrue(sess._config.gpu_options.allow_growth)

    def test_prepare_callbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = prepare_callbacks(tmpdir)

            # TensorBoard
            self.assertIsInstance(callbacks[0], TensorBoard)
            self.assertTrue(callbacks[0].log_dir.startswith(tmpdir))

            # CSVLogger
            self.assertIsInstance(callbacks[1], CSVLogger)
            self.assertTrue(callbacks[1].filename.startswith(tmpdir))

            # ModelCheckpoint
            self.assertIsInstance(callbacks[2], ModelCheckpoint)
            self.assertTrue(callbacks[2].filepath.startswith(tmpdir))

    def test_train_id_splitter(self):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                parser = argparse \
                    .ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
                output_path = tmpdir
                add_train_id_splitter_args(parser)
                arguments = "-i {} -o {} -e 1 --val-batch-size 10 --batch-size 10 --devices -1 " \
                            "--samples-before-report 20 cnn".format(IDENTIFIERS, output_path)
                args = parser.parse_args(arguments.split())

                train_id_splitter(args, model=build_cnn_from_args)
        except Exception as e:
            self.fail("cnn training raised {} with log {}".format(type(e), str(e)))

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                parser = argparse \
                    .ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
                output_path = tmpdir
                add_train_id_splitter_args(parser)
                arguments = "-i {} -o {} -e 1 --val-batch-size 10 --batch-size 10 --devices -1 " \
                            "--samples-before-report 20 rnn".format(IDENTIFIERS, output_path)
                args = parser.parse_args(arguments.split())

                train_id_splitter(args, model=build_rnn_from_args)
        except Exception as e:
            self.fail("rnn training raised {} with log {}".format(type(e), str(e)))
