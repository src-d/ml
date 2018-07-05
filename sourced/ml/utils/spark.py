import logging
import os
import sys
import pkg_resources
import zipfile

os.environ["PYSPARK_PYTHON"] = sys.executable

import pyspark  # noqa
from pyspark.sql import SparkSession  # noqa


class SparkDefault:
    """
    Default arguments for create_spark function and __main__
    """
    MASTER_ADDRESS = "local[*]"
    LOCAL_DIR = "/tmp/spark"
    LOG_LEVEL = "WARN"
    CONFIG = (
        # to clean up unpacked Siva files, see https://github.com/src-d/engine/issues/348
        "spark.tech.sourced.engine.cleanup.skip=false",
        # to skip broken siva files
        "spark.tech.sourced.engine.skip.read.errors=true",
    )
    JAR_PACKAGES = tuple()
    MEMORY = ""
    STORAGE_LEVEL = "MEMORY_AND_DISK"
    DEP_ZIP = False


def add_spark_args(my_parser, default_packages=None):
    my_parser.add_argument(
        "-s", "--spark", default=SparkDefault.MASTER_ADDRESS,
        help="Spark's master address.")
    my_parser.add_argument(
        "--config", nargs="+", default=SparkDefault.CONFIG,
        help="Spark configuration (key=value).")
    my_parser.add_argument(
        "-m", "--memory", default=SparkDefault.MEMORY,
        help="Handy memory config for spark. -m 4G,10G,2G is equivalent to "
             "--config spark.executor.memory=4G "
             "--config spark.driver.memory=10G "
             "--config spark.driver.maxResultSize=2G."
             "Numbers are floats separated by commas.")
    if default_packages is None:
        default_packages = SparkDefault.JAR_PACKAGES
    my_parser.add_argument(
        "--package", nargs="+", default=default_packages, dest="packages",
        help="Additional Spark packages.")
    my_parser.add_argument(
        "--spark-local-dir", default=SparkDefault.LOCAL_DIR,
        help="Spark local directory.")
    spark_log_levels = ("ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN")
    my_parser.add_argument("--spark-log-level", default=SparkDefault.LOG_LEVEL,
                           choices=spark_log_levels, help="Spark log level")
    my_parser.add_argument("--dep-zip", default=SparkDefault.DEP_ZIP,
                           action="store_true", help="Adds ml and engine to the spark context.")
    persistences = [att for att in pyspark.StorageLevel.__dict__.keys() if "__" not in att]
    my_parser.add_argument(
        "--persist", default=SparkDefault.STORAGE_LEVEL, choices=persistences,
        help="Spark persistence type (StorageLevel.*).")
    my_parser.add_argument(
        "--pause", action="store_true",
        help="Do not terminate in the end - useful for inspecting Spark Web UI.")
    my_parser.add_argument("--explain", action="store_true",
                           help="Print the PySpark execution plans.")


def create_spark(session_name,
                 spark=SparkDefault.MASTER_ADDRESS,
                 spark_local_dir=SparkDefault.LOCAL_DIR,
                 config=SparkDefault.CONFIG,
                 packages=SparkDefault.JAR_PACKAGES,
                 spark_log_level=SparkDefault.LOG_LEVEL,
                 memory=SparkDefault.MEMORY,
                 dep_zip=SparkDefault.DEP_ZIP):
    log = logging.getLogger("spark")
    log.info("Starting %s on %s", session_name, spark)
    # Hide py4j verbose logging (It appears in travis mostly)
    logging.getLogger("py4j").setLevel(logging.WARNING)
    config += get_spark_memory_config(memory)
    builder = SparkSession.builder.master(spark).appName(session_name)
    builder = builder.config("spark.jars.packages", ",".join(packages))
    builder = builder.config("spark.local.dir", spark_local_dir)
    for cfg in config:
        builder = builder.config(*cfg.split("=", 1))
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(spark_log_level)
    if dep_zip:
        zip_path = zip_sourced()
        session.sparkContext.addPyFile(zip_path)
    return session


def zip_sourced():
    zip_path = os.path.expanduser("~/.cache/sourced/ml/sourced.zip")
    if not os.path.exists(zip_path):
        os.makedirs(zip_path.split("sourced.zip")[0])
        working_dir = os.getcwd()
        os.chdir(pkg_resources.working_set.by_key["sourced-ml"].location)
        with zipfile.PyZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write("sourced")
            zf.writepy("sourced/ml", "sourced")
            zf.writepy("sourced/engine", "sourced")
        os.chdir(working_dir)
    return zip_path


def get_spark_memory_config(memory=SparkDefault.MEMORY):
    """
    Assemble memory configuration for a Spark session
    :param memory: string with memory configuration for spark
    :return: memory configuration
    """
    if not memory:
        return tuple()
    memory = memory.split(",")
    if len(memory) != 3:
        raise ValueError("Expected 3 memory parameters but got %s. Please check --help "
                         "to see how -m/--memory should be used." % len(memory))
    return ("spark.executor.memory=" + memory[0],
            "spark.driver.memory=" + memory[1],
            "spark.driver.maxResultSize=" + memory[2])
