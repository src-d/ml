import logging
from typing import Union

from pyspark import RDD, Row, StorageLevel
from pyspark.sql import DataFrame, functions

from sourced.ml.extractors.helpers import filter_kwargs
from sourced.ml.transformers.transformer import Transformer
from sourced.ml.transformers.uast2bag_features import Uast2BagFeatures
from sourced.ml.utils import EngineConstants, assemble_spark_config, create_engine, create_spark, \
    SparkDefault


class Repartitioner(Transformer):
    def __init__(self, partitions: int, shuffle: bool=False, **kwargs):
        super().__init__(**kwargs)
        self.partitions = partitions
        self.shuffle = shuffle

    def __call__(self, head: RDD):
        return head.coalesce(self.partitions, self.shuffle)

    @staticmethod
    def maybe(partitions: Union[int, None], shuffle: bool=False, multiplier: int=1):
        if partitions is not None:
            return Repartitioner(partitions * multiplier, shuffle)
        else:
            return Identity()


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


class Ignition(Transformer):
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    def __getstate__(self):
        state = super().__getstate__()
        del state["engine"]
        return state

    def __call__(self, _):
        return self.engine


class DzhigurdaFiles(Transformer):
    def __init__(self, dzhigurda, **kwargs):
        super().__init__(**kwargs)
        self.dzhigurda = dzhigurda

    def __call__(self, engine):
        head_ref = engine.repositories.references.head_ref
        if self.dzhigurda < 0:
            # Use all available commits
            chosen = head_ref.all_reference_commits
        elif self.dzhigurda == 0:
            # Use only the first commit on a reference.
            chosen = head_ref.commits
        else:
            commits = head_ref.all_reference_commits
            chosen = commits.filter(commits.index <= self.dzhigurda)
        return chosen.tree_entries.blobs


class HeadFiles(Transformer):
    def __call__(self, engine):
        return engine.repositories.references.head_ref.commits.tree_entries.blobs


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


class LanguageSelector(Transformer):
    def __init__(self, languages: Union[list, tuple], blacklist=False, **kwargs):
        super().__init__(**kwargs)
        self.languages = languages
        self.blacklist = blacklist

    def __call__(self, files: DataFrame) -> DataFrame:
        files = files.dropDuplicates(("blob_id",)).filter("is_binary = 'false'")
        classified = files.classify_languages()
        if not self.blacklist:
            lang_filter = classified.lang == self.languages[0]
            for lang in self.languages[1:]:
                lang_filter |= classified.lang == lang
        else:
            lang_filter = classified.lang != self.languages[0]
            for lang in self.languages[1:]:
                lang_filter &= classified.lang != lang
        filtered_by_lang = classified.filter(lang_filter)
        return filtered_by_lang


class UastExtractor(Transformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, files: DataFrame) -> DataFrame:
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
        if not row[EngineConstants.Columns.Uast]:
            return
        row_dict = row.asDict()
        row_dict[EngineConstants.Columns.Uast] = []
        for i, uast in enumerate(row[EngineConstants.Columns.Uast]):
            try:
                row_dict[EngineConstants.Columns.Uast].append(self.parse_uast(uast))
            except:  # nopep8
                self._log.error("\nBabelfish Error: Failed to parse uast for document %s for uast "
                                "#%s" % (row[Uast2BagFeatures.Columns.document], i))
        yield Row(**row_dict)


def create_parquet_loader(session_name, repositories,
                          config=SparkDefault.CONFIG,
                          packages=SparkDefault.PACKAGES,
                          spark=SparkDefault.MASTER_ADDRESS,
                          spark_local_dir=SparkDefault.LOCAL_DIR,
                          spark_log_level=SparkDefault.LOG_LEVEL,
                          memory=SparkDefault.MEMORY,
                          dep_zip=False):
    config = assemble_spark_config(config=config, memory=memory)
    session = create_spark(session_name, spark=spark, spark_local_dir=spark_local_dir,
                           config=config, packages=packages, spark_log_level=spark_log_level,
                           dep_zip=dep_zip)
    log = logging.getLogger("parquet")
    log.info("Initializing on %s", repositories)
    parquet = ParquetLoader(session, repositories)
    return parquet


def create_uast_source(args, session_name, select=HeadFiles, language_selector=None,
                       extract_uast=True):
    if args.parquet:
        parquet_loader_args = filter_kwargs(args.__dict__, create_parquet_loader)
        start_point = create_parquet_loader(session_name, **parquet_loader_args)
        root = start_point
        if extract_uast and "uast" not in [col.name for col in start_point.execute().schema]:
            raise ValueError("The parquet files do not contain UASTs.")
    else:
        engine_args = filter_kwargs(args.__dict__, create_engine)
        root = create_engine(session_name, **engine_args)
        if language_selector is None:
            language_selector = LanguageSelector(languages=args.languages)
        start_point = Ignition(root, explain=args.explain) \
            .link(select()) \
            .link(language_selector)
        if extract_uast:
            start_point = start_point.link(UastExtractor())
    return root, start_point
