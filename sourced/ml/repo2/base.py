import importlib
from typing import Union

from pyspark import StorageLevel
from pyspark.sql.types import Row

from sourced.ml.pickleable_logger import PickleableLogger  # nopep8


class Transformer(PickleableLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._children = []

    @property
    def children(self):
        return self._children

    def link(self, transformer):
        self.children.append(transformer)
        return transformer

    def unlink(self, transformer):
        self.children.remove(transformer)
        return self

    def execute(self, head):
        new_head = self(head)
        results = []
        for child in self.children:
            results.append(child.execute(new_head))
        return results

    def _get_log_name(self):
        return self.__class__.__name__

    def __call__(self, head):
        raise NotImplementedError()


class Collector(Transformer):
    def __call__(self, head):
        return head.collect()


class Cacher(Transformer):
    def __init__(self, persistence, **kwargs):
        super().__init__(**kwargs)
        self.persistence = getattr(StorageLevel, persistence)

    def __call__(self, head):
        return head.persist(self.persistence)


class UastExtractor(Transformer):
    def __init__(self, engine, languages: Union[list, tuple], **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.languages = languages

    def __getstate__(self):
        state = super().__getstate__()
        del state["engine"]
        return state

    def execute(self, files=None):
        if files is None:
            files = self.engine.repositories.references.head_ref.files
        return super().execute(files)

    def __call__(self, files):
        classified = files.classify_languages()
        lang_filter = classified.lang == self.languages[0]
        for lang in self.languages[1:]:
            lang_filter |= classified.lang == lang
        filtered_by_lang = classified.filter(lang_filter)
        uasts = filtered_by_lang.extract_uasts()
        return uasts


class UastDeserializer(Transformer):
    def __setstate__(self, state):
        super().__setstate__(state)
        from bblfsh.sdkversion import VERSION
        self.parse_uast = importlib.import_module(
            "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION
        ).Node.FromString

    def __call__(self, rows):
        return rows.rdd.flatMap(self.deserialize_uast)

    def deserialize_uast(self, row):
        if not row.uast:
            return
        row_dict = row.asDict()
        row_dict["uast"] = self.parse_uast(row.uast[0])
        yield Row(**row_dict)
