import sys

from modelforge.logs import setup_logging

from sourced.ml.utils import create_spark
from sourced.ml.utils.engine import add_engine_dependencies, add_bblfsh_dependencies, \
    get_engine_version


utmain = sys.modules['__main__']
if utmain.__package__ == "unittest" and utmain.__spec__ is None:
    from collections import namedtuple
    ModuleSpec = namedtuple("ModuleSpec", ["name"])
    utmain.__spec__ = ModuleSpec("unittest.__main__")
    del ModuleSpec
del utmain


def create_spark_for_test(name="test"):
    config = []
    packages = []
    bblfsh = "localhost"
    engine = get_engine_version()
    add_engine_dependencies(engine=engine, config=config, packages=packages)
    add_bblfsh_dependencies(bblfsh=bblfsh, config=config)
    return create_spark(name, config=config, packages=packages)


def setup():
    setup_logging("INFO")
