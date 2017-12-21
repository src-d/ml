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
