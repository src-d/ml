import logging
import os
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable

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
    PACKAGE = []


def add_spark_args(my_parser):
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
    my_parser.add_argument(
        "--package", nargs="+", default=SparkDefault.PACKAGE,
        help="Additional Spark package.")
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

def add_engine_args(my_parser):
    add_spark_args(my_parser)
    my_parser.add_argument(
        "--bblfsh", default=EngineDefault.BBLFSH,
        help="Babelfish server's address.")
    my_parser.add_argument(
        "--engine", default=EngineDefault.VERSION,
        help="source{d} engine version.")
    my_parser.add_argument("--explain", action="store_true",
                           help="Print the PySpark execution plans.")


def create_spark(session_name,
                 spark=SparkDefault.MASTER_ADDRESS,
                 spark_local_dir=SparkDefault.LOCAL_DIR,
                 config=SparkDefault.CONFIG,
                 package=SparkDefault.PACKAGE,
                 spark_log_level=SparkDefault.LOG_LEVEL,
                 **kwargs):
    log = logging.getLogger("spark")
    log.info("Starting %s on %s", session_name, spark)
    builder = SparkSession.builder.master(spark).appName(session_name)
    builder = builder.config(
        "spark.jars.packages", ",".join(package))
    builder = builder.config("spark.local.dir", spark_local_dir)
    for cfg in config:
        builder = builder.config(*cfg.split("=", 1))
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(spark_log_level)
    return session


class EngineDefault:
    """
    Default arguments for create_engine function and __main__
    """
    BBLFSH = "localhost"
    VERSION = "0.3.1"


def create_engine(session_name, repositories,
                  bblfsh=EngineDefault.BBLFSH,
                  engine=EngineDefault.VERSION,
                  **kwargs):
    kwargs["config"].append("spark.tech.sourced.bblfsh.grpc.host=" + bblfsh)
    # TODO(vmarkovtsev): figure out why is this option needed
    kwargs["config"].append("spark.tech.sourced.engine.cleanup.skip=true")
    kwargs["package"].append("tech.sourced:engine:" + engine)
    if kwargs.get("memory", None) is not None:
        memory = kwargs["memory"].split(",")
        kwargs["memory"].append("spark.executor.memory=%s" + memory[0])
        kwargs["memory"].append("spark.driver.memory=%s" + memory[1])
        kwargs["memory"].append("spark.driver.maxResultSize=%s" + memory[2])
    session = create_spark(session_name, **kwargs)
    log = logging.getLogger("engine")
    log.info("Initializing on %s", repositories)
    try:
        engine = Engine(session, repositories, "siva")
    except:
        engine = Engine(session, repositories)
    return engine
