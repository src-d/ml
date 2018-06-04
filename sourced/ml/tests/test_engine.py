import unittest

from sourced.ml.utils.engine import add_bblfsh_dependencies, add_engine_dependencies


class EngineTests(unittest.TestCase):
    def test_bblfsh_dependencies(self):
        config = []
        add_bblfsh_dependencies("localhost", config)
        self.assertEqual(config, ["spark.tech.sourced.bblfsh.grpc.host=localhost"])

    def test_engine_dependencies(self):
        config = []
        packages = []
        add_engine_dependencies("latest", config, packages)
        self.assertEqual(config, ["spark.tech.sourced.engine.cleanup.skip=false",
                                  "spark.tech.sourced.engine.skip.read.errors=true"])
        self.assertEqual(packages, ["tech.sourced:engine:latest"])
