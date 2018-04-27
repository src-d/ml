import functools
import logging
import requests
from pkg_resources import get_distribution, DistributionNotFound
from sourced.engine import Engine
from sourced.ml.utils.spark import add_spark_args, assemble_spark_config, create_spark


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
    my_parser.add_argument("--bblfsh", default=None,
                           help="Babelfish server's address.")
    my_parser.add_argument("--engine", default=None,
                           help="source{d} engine version.")
    my_parser.add_argument("--repository-format", default="siva",
                           help="Repository storage input format.")
    my_parser.add_argument("--explain", action="store_true",
                           help="Print the PySpark execution plans.")


def add_engine_dependencies(engine, config=None, packages=None):
    # to clean up unpacked Siva files, see https://github.com/src-d/engine/issues/348
    config.append("spark.tech.sourced.engine.cleanup.skip=false")
    packages.append("tech.sourced:engine:" + engine)


def add_bblfsh_dependencies(bblfsh, config=None):
    config.append("spark.tech.sourced.bblfsh.grpc.host=" + bblfsh)


def create_engine(session_name, repositories,
                  bblfsh=None,
                  engine=None,
                  config=None, packages=None, memory="",
                  repository_format="siva", **spark_kwargs):
    if not bblfsh:
        bblfsh = "localhost"
    if not engine:
        try:
            engine = get_distribution("sourced-engine").version
        except DistributionNotFound:
            log = logging.getLogger("engine_version")
            engine = requests.get("https://api.github.com/repos/src-d/engine/releases/latest") \
                .json()["tag_name"].replace("v", "")
            log.warning("Engine not found, queried GitHub to get the latest release tag (%s)",
                        engine)
    config, packages = assemble_spark_config(config=config, packages=packages, memory=memory)
    add_engine_dependencies(engine=engine, config=config, packages=packages)
    add_bblfsh_dependencies(bblfsh=bblfsh, config=config)
    session = create_spark(session_name, config=config, packages=packages, **spark_kwargs)
    log = logging.getLogger("engine")
    log.info("Initializing on %s", repositories)
    engine = Engine(session, repositories, repository_format)
    return engine


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
