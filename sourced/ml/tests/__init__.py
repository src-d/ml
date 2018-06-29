import sys

from modelforge.logs import setup_logging

from sourced.ml.utils import create_spark
from sourced.ml.utils.engine import get_engine_package, get_bblfsh_dependency, \
    get_engine_version


utmain = sys.modules['__main__']
if utmain.__package__ == "unittest" and utmain.__spec__ is None:
    from collections import namedtuple
    ModuleSpec = namedtuple("ModuleSpec", ["name"])
    utmain.__spec__ = ModuleSpec("unittest.__main__")
    del ModuleSpec
del utmain


def create_spark_for_test(name="test"):
    packages = (get_engine_package(get_engine_version()),)
    config = (get_bblfsh_dependency("localhost"),)
    return create_spark(name, config=config, packages=packages)


def setup():
    setup_logging("INFO")
