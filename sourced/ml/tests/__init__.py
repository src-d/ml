import sys
from unittest import SkipTest

from modelforge import slogging

from sourced.ml.utils.engine import (
    get_bblfsh_dependency, get_engine_package, get_engine_version)
from sourced.ml.utils.spark import create_spark


utmain = sys.modules["__main__"]
if utmain.__package__ == "unittest" and utmain.__spec__ is None:
    from collections import namedtuple
    ModuleSpec = namedtuple("ModuleSpec", ["name"])
    utmain.__spec__ = ModuleSpec("unittest.__main__")
    del ModuleSpec
del utmain


def create_spark_for_test(name="test"):
    if sys.version_info >= (3, 7):
        raise SkipTest("Python 3.7 is not yet supported.")
    packages = (get_engine_package(get_engine_version()),)
    config = (get_bblfsh_dependency("localhost"),)
    return create_spark(name, config=config, packages=packages)


def has_tensorflow():
    try:
        import tensorflow  # noqa
        return True
    except ImportError:
        return False


def setup():
    slogging.setup("INFO", False)
