import argparse
import logging
from typing import List, Union

from sourced.engine.engine import BlobsDataFrame, BlobsWithLanguageDataFrame
from pyspark import RDD, Row, StorageLevel
from pyspark.sql import DataFrame, functions

from sourced.ml.extractors.helpers import filter_kwargs
from sourced.ml.transformers.transformer import Transformer
from sourced.ml.transformers.uast2bag_features import Uast2BagFeatures
from sourced.ml.utils import EngineConstants, get_spark_memory_config, create_engine, \
    create_spark, SparkDefault


class Repartitioner(Transformer):
    """
    Repartitioner uses one of the three ways to split an RDD into partitions:
    1. coalesce() if shuffle=False, keymap=None
    2. repartition() if shuffle=True, keymap=None
    3. partitionBy() if keymap is not None
    """
    def __init__(self, partitions: int, shuffle: bool=False, keymap: callable=None, **kwargs):
        super().__init__(**kwargs)
        self.partitions = partitions
        self.shuffle = shuffle
        self.keymap = keymap

    def __call__(self, head: RDD):
        if self.keymap is None:
            return head.coalesce(self.partitions, self.shuffle)
        # partitionBy the key extracted using self.keymap
        try:
            # this checks if keymap is an identity
            probe = self.keymap("probe")
        except:  # noqa: E722
            probe = None
        if probe != "probe":
            head = head.map(lambda x: (self.keymap(x), x))
        return head \
            .partitionBy(self.partitions) \
            .map(lambda x: x[1])

    @staticmethod
    def maybe(partitions: Union[int, None], shuffle: bool=False, keymap: callable=None,
              multiplier: int=1):
        if partitions is not None:
            return Repartitioner(partitions * multiplier, shuffle=shuffle, keymap=keymap)
        else:
            return Identity()


class PartitionSelector(Transformer):
    """
    PartitionSelector return the partition by specific index.
    """
    def __init__(self, partition_index: int, **kwargs):
        super().__init__(**kwargs)
        self.partition_index = partition_index

    def __call__(self, head: RDD):
        index = self.partition_index

        def partition_filter(split_index, part):
            if split_index == index:
                for row in part:
                    yield row

        return head.mapPartitionsWithIndex(partition_filter, True)


class CsvSaver(Transformer):
    def __init__(self, output: str, **kwargs):
        super().__init__(**kwargs)
        self.output = output

    def __call__(self, head: RDD):
        self._log.info("Writing %s", self.output)
        return head.toDF() \
            .coalesce(1) \
            .write \
            .option("header", "true") \
            .mode("overwrite") \
            .csv(self.output)


class Rower(Transformer):
    def __init__(self, dicter, **kwargs):
        super().__init__(**kwargs)
        self.dicter = dicter

    def __call__(self, head: RDD):
        return head.map(lambda x: Row(**self.dicter(x)))


class Sampler(Transformer):
    """
    Wraps `sample()` function from pyspark Dataframe.
    """
    def __init__(self, with_replacement=False, fraction=0.05, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.with_replacement = with_replacement
        self.fraction = fraction
        self.seed = seed

    def __call__(self, head: RDD):
        return head.sample(self.with_replacement, self.fraction, self.seed)


class Collector(Transformer):
    def __call__(self, head: RDD):
        return head.collect()


class Distinct(Transformer):
    def __call__(self, head: RDD):
        return head.distinct()


class First(Transformer):
    def __call__(self, head: RDD):
        return head.first()


class Identity(Transformer):
    def __call__(self, head: RDD):
        return head


class Cacher(Transformer):
    def __init__(self, persistence, **kwargs):
        super().__init__(**kwargs)
        self.persistence = getattr(StorageLevel, persistence)
        self.head = None
        self.trace = None

    def __getstate__(self):
        state = super().__getstate__()
        state["head"] = None
        state["trace"] = None
        return state

    def __call__(self, head: RDD):
        if self.head is None or self.trace != self.path():
            self.head = head.persist(self.persistence)
            self.trace = self.path()
        return self.head

    @staticmethod
    def maybe(persistence):
        if persistence is not None:
            return Cacher(persistence)
        else:
            return Identity()

    def unpersist(self):
        self.head.unpersist()


class Ignition(Transformer):
    """
    All pipelines start with this transformer - it returns all the repositories from the engine.
    """
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    def __getstate__(self):
        state = super().__getstate__()
        del state["engine"]
        return state

    def __call__(self, _) -> DataFrame:
        return self.engine.repositories


class RepositoriesFilter(Transformer):
    """
    Filters repositories by a regular expression over the identifiers
    (ex. "github.com/src-d/vecino" or "file:///tmp/vecino-yzv92l0i/repo").
    """
    def __init__(self, idre: str, **kwargs):
        super().__init__(**kwargs)
        self.idre = idre

    def __call__(self, repositories: DataFrame) -> DataFrame:
        return repositories.filter(repositories["id"].rlike(self.idre))


class DzhigurdaFiles(Transformer):
    def __init__(self, dzhigurda, **kwargs):
        super().__init__(**kwargs)
        self.dzhigurda = dzhigurda

    def __call__(self, repositories: DataFrame) -> DataFrame:
        head_ref = repositories.references.head_ref
        if self.dzhigurda < 0:
            # Use all available commits
            chosen = head_ref.all_reference_commits
        elif self.dzhigurda == 0:
            # Use only the first commit on a reference.
            # This case is completely the same with HeadFiles Transformer
            chosen = head_ref.commits
        else:
            commits = head_ref.all_reference_commits
            chosen = commits.filter(commits.index <= self.dzhigurda)
        return chosen.tree_entries.blobs


class HeadFiles(Transformer):
    def __call__(self, repositories: DataFrame) -> DataFrame:
        return repositories.references.head_ref.commits.tree_entries.blobs


class Counter(Transformer):
    def __init__(self, distinct=False, approximate=False, **kwargs):
        super().__init__(**kwargs)
        self.distinct = distinct
        self.approximate = approximate

    def __call__(self, head: RDD):
        if self.distinct and not self.approximate:
            head = head.distinct()
        if self.explained:
            self._log.info("toDebugString():\n%s", head.toDebugString().decode())
        if not self.approximate or not self.distinct:
            return head.count()
        return head.countApproxDistinct()


class LanguageExtractor(Transformer):
    def __call__(self, files: BlobsDataFrame) -> DataFrame:
        if not isinstance(files, BlobsDataFrame):
            raise TypeError("Argument type is not BlobsDataFrame. "
                            "Language extraction can not be performed.")
        return files \
            .dropDuplicates(("blob_id",)) \
            .filter("is_binary = 'false'") \
            .classify_languages()


class LanguageSelector(Transformer):
    def __init__(self, languages: List[str], blacklist=False, **kwargs):
        super().__init__(**kwargs)
        self.languages = languages
        self.blacklist = blacklist

    def __call__(self, files: BlobsWithLanguageDataFrame) -> DataFrame:
        return files[files.lang.isin(self.languages) != self.blacklist]

    @staticmethod
    def maybe(languages, blacklist):
        if languages is None:
            return Identity()
        return LanguageSelector(languages, blacklist)


class UastExtractor(Transformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, files: Union[BlobsDataFrame, BlobsWithLanguageDataFrame]) -> DataFrame:
        if not isinstance(files, (BlobsDataFrame, BlobsWithLanguageDataFrame)):
            raise TypeError("Argument type should be BlobsDataFrame or BlobsWithLanguageDataFrame,"
                            " got %s" % type(files).__name__)
        # if UAST is not extracted, returns an empty list that we filter out here
        return files.extract_uasts().where(functions.size(functions.col("uast")) > 0)


class FieldsSelector(Transformer):
    def __init__(self, fields: Union[list, tuple], **kwargs):
        super().__init__(**kwargs)
        self.fields = fields

    def __call__(self, rdd: RDD) -> RDD:
        def select_fields(row):
            return Row(**{f: getattr(row, f) for f in self.fields})
        res = rdd.map(select_fields)
        if self.explained:
            self._log.info("toDebugString():\n%s", res.toDebugString().decode())
        return res


class ParquetSaver(Transformer):
    def __init__(self, save_loc, **kwargs):
        super().__init__(**kwargs)
        self.save_loc = save_loc

    def __call__(self, rdd: RDD):
        if self.explained:
            self._log.info("toDebugString():\n%s", rdd.toDebugString().decode())
        rdd.toDF().write.parquet(self.save_loc)


class ParquetLoader(Transformer):
    def __init__(self, session, paths, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.paths = paths

    def __getstate__(self):
        state = super().__getstate__()
        del state["session"]
        return state

    def __call__(self, _):
        if isinstance(self.paths, (list, tuple)):
            return self.session.read.parquet(*self.paths)
        if isinstance(self.paths, str):
            return self.session.read.parquet(self.paths)
        raise ValueError


class UastDeserializer(Transformer):
    def __setstate__(self, state):
        super().__setstate__(state)
        from bblfsh import Node
        self.parse_uast = Node.FromString

    def __call__(self, rows: RDD) -> RDD:
        return rows.flatMap(self.deserialize_uast)

    def deserialize_uast(self, row: Row):
        if EngineConstants.Columns.Uast not in row:
            return
        if not row[EngineConstants.Columns.Uast]:
            return
        row_dict = row.asDict()
        row_dict[EngineConstants.Columns.Uast] = []
        for i, uast in enumerate(row[EngineConstants.Columns.Uast]):
            try:
                row_dict[EngineConstants.Columns.Uast].append(self.parse_uast(uast))
            except:  # noqa
                self._log.error("\nBabelfish Error: Failed to parse uast for document %s for uast "
                                "#%s" % (row[Uast2BagFeatures.Columns.document], i))
        yield Row(**row_dict)


def create_parquet_loader(session_name, repositories,
                          config=SparkDefault.CONFIG,
                          packages=SparkDefault.JAR_PACKAGES,
                          spark=SparkDefault.MASTER_ADDRESS,
                          spark_local_dir=SparkDefault.LOCAL_DIR,
                          spark_log_level=SparkDefault.LOG_LEVEL,
                          memory=SparkDefault.MEMORY,
                          dep_zip=SparkDefault.DEP_ZIP):
    config += get_spark_memory_config(memory)
    session = create_spark(session_name, spark=spark, spark_local_dir=spark_local_dir,
                           config=config, packages=packages, spark_log_level=spark_log_level,
                           dep_zip=dep_zip)
    log = logging.getLogger("parquet")
    log.info("Initializing on %s", repositories)
    parquet = ParquetLoader(session, repositories)
    return parquet


def create_file_source(args: argparse.Namespace, session_name: str):
    if args.parquet:
        parquet_loader_args = filter_kwargs(args.__dict__, create_parquet_loader)
        root = create_parquet_loader(session_name, **parquet_loader_args)
        file_source = root.link(LanguageSelector.maybe(languages=args.languages,
                                                       blacklist=args.blacklist))
    else:
        engine_args = filter_kwargs(args.__dict__, create_engine)
        root = Ignition(create_engine(session_name, **engine_args), explain=args.explain)
        file_source = root.link(DzhigurdaFiles(args.dzhigurda))
        if args.languages is not None:
            file_source = file_source \
                .link(LanguageExtractor()) \
                .link(LanguageSelector(languages=args.languages, blacklist=args.blacklist))

    return root, file_source


def create_uast_source(args: argparse.Namespace, session_name: str):
    root, file_source = create_file_source(args, session_name)
    if args.parquet:
        # Assume that we already have uast column inside, because we cannot convert parquet files
        # back to sourced-engine format to extract UASTs.
        return root, file_source
    return root, file_source.link(UastExtractor())
