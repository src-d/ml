from io import StringIO
from typing import Union, Tuple

from sourced.ml.utils import PickleableLogger  # nopep8


class Transformer(PickleableLogger):
    BLOCKS = False  # set to True if __call__ launches PySpark

    def __init__(self, explain=None, **kwargs):
        super().__init__(**kwargs)
        self._children = []
        self._parent = None
        self._explained = explain

    def __getstate__(self):
        state = super().__getstate__()
        del state["_parent"]
        del state["_children"]
        return state

    @property
    def explained(self):
        if self._explained is None and self.parent is not None:
            return self.parent.explained
        return bool(self._explained)

    @property
    def children(self):
        return tuple(self._children)

    @property
    def parent(self):
        return self._parent

    def path(self):
        node = self
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def link(self, *transformer: "Transformer") -> Union[Tuple["Transformer"], "Transformer"]:
        def link_one(t: Transformer) -> Transformer:
            self._children.append(t)
            t._parent = self
            return t

        if len(transformer) == 1:
            return link_one(transformer[0])
        return tuple(link_one(t) for t in transformer)

    def unlink(self, *transformer: "Transformer"):
        for t in transformer:
            self._children.remove(t)
            t._parent = None
        return self

    def __rshift__(self, other):
        """Shortcut for link"""
        if isinstance(other, (list, tuple)):
            return self.link(*other)
        return self.link(other)

    def __lshift__(self, other):
        """Shortcut for unlink"""
        if isinstance(other, (list, tuple)):
            return self.unlink(*other)
        return self.unlink(other)

    def _explode(self, head, context):
        if context[-1] is not self:
            context.append(self)
            if not self._children or self.BLOCKS:
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
        head = self.execute(head)
        pipeline = [self]
        node = self
        while node.parent is not None:
            node = node.parent
            pipeline.append(node)
        pipeline.reverse()
        return self._explode(head, pipeline)

    def execute(self, head=None):
        """
        Execute the node together with all its dependencies, in order.
        :param head: The input to feed to the ultimate parent. Can be None - the default one \
                     will be used if possible.
        :return: The result of the execution.
        """
        pipeline = self.path()
        if not self._children:
            self._log.info(self._format_pipeline(pipeline))
        for node in pipeline:
            head = node(head)
        return head

    def graph(self, name="source-d", stream=None):
        if stream is None:
            stream = StringIO()
        stream.write("digraph %s {\n" % name)
        counters = {}
        nodes = {}
        queue = [(self, None)]
        while queue:
            node, parent = queue.pop(0)
            try:
                myself = nodes[node]
            except KeyError:
                index = counters.setdefault(type(node), 0)
                index += 1
                counters[type(node)] = index
                myself = "%s %d" % (type(node).__name__, index)
                nodes[node] = myself
            if parent is not None:
                stream.write("\t\"%s\" -> \"%s\"\n" % (parent, myself))
            for child in node._children:
                queue.append((child, myself))
        stream.write("}\n")
        return stream

    def _get_log_name(self):
        return self.__class__.__name__

    @staticmethod
    def _format_pipeline(pipeline):
        return " -> ".join(type(n).__name__ for n in pipeline)

    def __call__(self, head):
        raise NotImplementedError()


class Execute(Transformer):
    """
    Special transformer to execute all the pipeline.
    As soon as one links anything to this Transformer it call execute() for the pipeline.
    It is not possible to link anything to this transformer.
    """

    def __init__(self, head=None, explain=None, **kwargs):
        super().__init__(explain, **kwargs)
        self.head = head
        self._real_parent = None

    @property
    def _parent(self):
        return self._real_parent

    @_parent.setter
    def _parent(self, value: Transformer):
        self._real_parent = value
        if value is not None:
            value.execute(self.head)

    def link(self, *transformer: "Transformer") -> Union[Tuple["Transformer"], "Transformer"]:
        raise AssertionError("It is not possible to link anything after Leaf.")

    def __call__(self, head):
        return head
