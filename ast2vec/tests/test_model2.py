import os
import unittest

from ast2vec.model2.base import Model2Base


class FromModel:
    NAME = "from"

    def load(self, source):
        pass


class ToModel:
    NAME = "to"

    def save(self, output, deps=None):
        pass


class Model2Test(Model2Base):
    MODEL_FROM_CLASS = FromModel
    MODEL_TO_CLASS = ToModel

    def convert_model(self, model):
        return ToModel()


class Model2BaseTests(unittest.TestCase):
    def test_convert(self):
        converter = Model2Test(num_processes=2)
        status = converter.convert(os.path.dirname(__file__), "xxx", pattern="**/*.py")
        self.assertGreater(status, 20)

if __name__ == "__main__":
    unittest.main()
