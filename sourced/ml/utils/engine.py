import logging
import os
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable

import pyspark  # nopep8
from pyspark.sql import SparkSession  # nopep8
from sourced.engine import Engine  # nopep8


class SparkDefault:
    """
    Default arguments for create_spark function and __main__
    """
    MASTER_ADDRESS = "local[*]"
    LOCAL_DIR = "/tmp/spark"
    LOG_LEVEL = "WARN"
    CONFIG = []
    PACKAGES = []


def add_spark_args(my_parser, default_packages=None):
    my_parser.add_argument(
        "-s", "--spark", default=SparkDefault.MASTER_ADDRESS,
        help="Spark's master address.")
    my_parser.add_argument(
        "--config", nargs="+", default=SparkDefault.CONFIG,
        help="Spark configuration (key=value).")
    my_parser.add_argument(
        "-m", "--memory",
        help="Handy memory config for spark. -m 4G,10G,2G is equivalent to "
             "--config spark.executor.memory=4G "
             "--config spark.driver.memory=10G "
             "--config spark.driver.maxResultSize=2G."
             "Numbers are floats separated by commas.")
    if default_packages is None:
        default_packages = SparkDefault.PACKAGES
    my_parser.add_argument(
        "--package", nargs="+", default=default_packages, dest="packages",
        help="Additional Spark packages.")
    my_parser.add_argument(
        "--spark-local-dir", default=SparkDefault.LOCAL_DIR,
        help="Spark local directory.")
    my_parser.add_argument("--spark-log-level", default=SparkDefault.LOG_LEVEL, choices=(
        "ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"),
                           help="Spark log level")
    persistences = [att for att in pyspark.StorageLevel.__dict__.keys() if "__" not in att]
    my_parser.add_argument(
        "--persist", default=None, choices=persistences,
        help="Spark persistence type (StorageLevel.*).")


class EngineDefault:
    """
    Default arguments for create_engine function and __main__
    """
    BBLFSH = "localhost"
    VERSION = "0.3.1"


def add_engine_args(my_parser, default_packages=None):
    add_spark_args(my_parser, default_packages=default_packages)
    my_parser.add_argument(
        "--bblfsh", default=EngineDefault.BBLFSH,
        help="Babelfish server's address.")
    my_parser.add_argument(
        "--engine", default=EngineDefault.VERSION,
        help="source{d} engine version.")
    my_parser.add_argument("--repository-format", default="siva",
                           help="Repository storage input format.")
    my_parser.add_argument("--explain", action="store_true",
                           help="Print the PySpark execution plans.")


def create_spark(session_name,
                 spark=SparkDefault.MASTER_ADDRESS,
                 spark_local_dir=SparkDefault.LOCAL_DIR,
                 config=SparkDefault.CONFIG,
                 packages=SparkDefault.PACKAGES,
                 spark_log_level=SparkDefault.LOG_LEVEL,
                 **_):  # **kwargs are discarded for convenience
    log = logging.getLogger("spark")
    log.info("Starting %s on %s", session_name, spark)
    builder = SparkSession.builder.master(spark).appName(session_name)
    builder = builder.config(
        "spark.jars.packages", ",".join(packages))
    builder = builder.config("spark.local.dir", spark_local_dir)
    for cfg in config:
        builder = builder.config(*cfg.split("=", 1))
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(spark_log_level)
    return session


def create_engine(session_name, repositories,
                  bblfsh=EngineDefault.BBLFSH,
                  engine=EngineDefault.VERSION,
                  config=None, packages=None, memory="",
                  repository_format="siva", **spark_kwargs):
    if config is None:
        config = []
    if packages is None:
        packages = []
    config.append("spark.tech.sourced.bblfsh.grpc.host=" + bblfsh)
    # TODO(vmarkovtsev): figure out why is this option needed
    config.append("spark.tech.sourced.engine.cleanup.skip=true")
    packages.append("tech.sourced:engine:" + engine)
    if memory:
        memory = memory.split(",")
        memory.append("spark.executor.memory=" + memory[0])
        memory.append("spark.driver.memory=" + memory[1])
        memory.append("spark.driver.maxResultSize=" + memory[2])
    session = create_spark(session_name, config=config, packages=packages, memory=memory,
                           **spark_kwargs)
    log = logging.getLogger("engine")
    log.info("Initializing on %s", repositories)
    engine = Engine(session, repositories, repository_format)
    return engine
