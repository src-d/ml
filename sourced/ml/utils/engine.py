import functools
import logging
import requests
from pkg_resources import get_distribution, DistributionNotFound
from sourced.engine import Engine
from sourced.ml.utils import add_spark_args, create_spark, SparkDefault


def get_engine_version():
    try:
        engine = get_distribution("sourced-engine").version
    except DistributionNotFound:
        log = logging.getLogger("engine_version")
        engine = requests.get("https://api.github.com/repos/src-d/engine/releases/latest") \
            .json()["tag_name"].replace("v", "")
        log.warning("Engine not found, queried GitHub to get the latest release tag (%s)",
                    engine)
    return engine


class EngineDefault:
    """
    Engine default initialization parameters
    """
    BBLFSH = "localhost"
    VERSION = get_engine_version()
    REPOSITORY_FORMAT = "siva"


class EngineConstants:
    """
    Constants for Engine usage.
    """
    class Columns:
        """
        Column names constants.
        """
        RepositoryId = "repository_id"
        Path = "path"
        BlobId = "blob_id"
        Uast = "uast"


def add_engine_args(my_parser, default_packages=None):
    add_spark_args(my_parser, default_packages=default_packages)
    my_parser.add_argument("--bblfsh", default=EngineDefault.BBLFSH,
                           help="Babelfish server's address.")
    my_parser.add_argument("--engine", default=EngineDefault.VERSION,
                           help="source{d} engine version.")
    my_parser.add_argument("--repository-format", default=EngineDefault.REPOSITORY_FORMAT,
                           help="Repository storage input format.")


def get_engine_package(engine):
    return "tech.sourced:engine:" + engine


def get_bblfsh_dependency(bblfsh):
    return "spark.tech.sourced.bblfsh.grpc.host=" + bblfsh


def create_engine(session_name, repositories,
                  repository_format=EngineDefault.REPOSITORY_FORMAT,
                  bblfsh=EngineDefault.BBLFSH,
                  engine=EngineDefault.VERSION,
                  config=SparkDefault.CONFIG,
                  packages=SparkDefault.JAR_PACKAGES,
                  spark=SparkDefault.MASTER_ADDRESS,
                  spark_local_dir=SparkDefault.LOCAL_DIR,
                  spark_log_level=SparkDefault.LOG_LEVEL,
                  dep_zip=SparkDefault.DEP_ZIP,
                  memory=SparkDefault.MEMORY):

    config += (get_bblfsh_dependency(bblfsh),)
    packages += (get_engine_package(engine),)
    session = create_spark(session_name, spark=spark, spark_local_dir=spark_local_dir,
                           config=config, packages=packages, spark_log_level=spark_log_level,
                           dep_zip=dep_zip, memory=memory)
    logging.getLogger("engine").info("Initializing engine on %s", repositories)
    return Engine(session, repositories, repository_format)


def pause(func):
    @functools.wraps(func)
    def wrapped_pause(cmdline_args, *args, **kwargs):
        try:
            return func(cmdline_args, *args, **kwargs)
        finally:
            if cmdline_args.pause:
                input("Press Enter to exit...")

    return wrapped_pause


def pipeline_graph(args, log, root):
    if args.graph:
        log.info("Dumping the graph to %s", args.graph)
        with open(args.graph, "w") as f:
            root.graph(stream=f)
