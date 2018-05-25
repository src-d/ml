import unittest

from sourced.ml.algorithms.id_splitter.nn_model import register_metric, METRICS


class IdSplitterMetrics(unittest.TestCase):
    def test_register_metric(self):
        fake_metric = "fake metric"
        register_metric(fake_metric)
        self.assertTrue(fake_metric in METRICS)
        METRICS.pop()
        self.assertFalse(fake_metric in METRICS)

    def test_raise_register_metric(self):
        bad_metric = 1
        with self.assertRaises(AssertionError):
            register_metric(bad_metric)
        self.assertTrue(bad_metric not in METRICS)


if __name__ == "__main__":
    unittest.main()
