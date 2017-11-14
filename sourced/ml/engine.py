import logging
import os
import sys

from pyspark.sql import SparkSession
from sourced.engine import Engine


__packages__ = [
    "tech.sourced:engine:%s",
    "com.datastax.spark:spark-cassandra-connector_2.11:2.0.3"
]


def create_engine(session_name, repositories, kwargs):
    log = logging.getLogger("engine")
    os.putenv("PYSPARK_PYTHON", sys.executable)
    log.info("Starting %s on %s", session_name, kwargs.spark)
    builder = SparkSession.builder.master(kwargs.spark).appName(session_name)
    builder = builder.config(
        "spark.jars.packages", ",".join(__packages__) % kwargs.engine)
    builder = builder.config(
        "spark.tech.sourced.bblfsh.grpc.host", kwargs.bblfsh)
    # TODO(vmarkovtsev): figure out why is this option needed
    builder = builder.config(
        "spark.tech.sourced.engine.cleanup.skip", "true")
    builder = builder.config("spark.local.dir", kwargs.spark_local_dir)
    for cfg in kwargs.config:
        builder = builder.config(*cfg.split("=", 1))
    session = builder.getOrCreate()
    engine = Engine(session, repositories)
    return engine
