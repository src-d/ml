import importlib
from typing import Union

from pyspark import StorageLevel
from pyspark.sql.types import Row

from sourced.ml.pickleable_logger import PickleableLogger  # nopep8


class Transformer(PickleableLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._children = []
        self._parent = None

    @property
    def children(self):
        return tuple(self._children)

    @property
    def parent(self):
        return self._parent

    def link(self, transformer):
        self._children.append(transformer)
        transformer._parent = self
        return transformer

    def unlink(self, transformer):
        self._children.remove(transformer)
        transformer._parent = None
        return self

    def _explode(self, head, context):
        if context[-1] is not self:
            context.append(self)
            if not self._children:
                self._log.info(self._format_pipeline(context))
            head = self(head)
        results = []
        for child in self._children:
            results.extend(child._explode(head, context.copy()))
        else:
            results.append(head)
        return results

    def explode(self, head=None):
        """
        Execute all the branches in the tree going from the node. The node itself
        is execute()-ed.
        :param head: The input to feed to the root. Can be None - the default one \
                     will be used if possible.
        :return: The results from all the leaves.
        """
        if self.parent is not None:
            head = self.execute(head)
        pipeline = [type(self).__name__]
        node = self
        while node.parent is not None:
            node = node.parent
            pipeline.append(node)
        pipeline.reverse()
        return self._explode(head, pipeline)

    def execute(self, head=None):
        """
        Execute the node together will all its dependencies, in order.
        :param head: The input to feed to the ultimate parent. Can be None - the default one \
                     will be used if possible.
        :return: The result of the execution.
        """
        pipeline = [self]
        node = self
        while node.parent is not None:
            node = node.parent
            pipeline.append(node)
        pipeline.reverse()
        if not self._children:
            self._log.info(self._format_pipeline(pipeline))
        for node in pipeline:
            head = node(head)
        return head

    def _get_log_name(self):
        return self.__class__.__name__

    @staticmethod
    def _format_pipeline(pipeline):
        return " -> ".join(type(n).__name__ for n in pipeline)

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

    def __call__(self, files):
        if files is None:
            files = self.engine.repositories.references.head_ref.files
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
