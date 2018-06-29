import unittest

from sourced.ml.utils.engine import get_bblfsh_dependency, get_engine_package


class EngineTests(unittest.TestCase):
    def test_bblfsh_dependency(self):
        self.assertEqual(get_bblfsh_dependency("localhost"),
                         "spark.tech.sourced.bblfsh.grpc.host=localhost")

    def test_engine_dependencies(self):
        self.assertEqual(get_engine_package("latest"), "tech.sourced:engine:latest")
