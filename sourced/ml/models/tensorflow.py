from typing import List

from modelforge import Model, register_model
import numpy

from sourced.ml.models.license import DEFAULT_LICENSE


@register_model
class TensorFlowModel(Model):
    """
    TensorFlow Protobuf model exported in the Modelforge format with GraphDef inside.
    """
    NAME = "tensorflow-model"
    VENDOR = "source{d}"
    DESCRIPTION = "TensorFlow Protobuf model that contains a GraphDef instance."
    LICENSE = DEFAULT_LICENSE

    def construct(self, graphdef: "tensorflow.GraphDef" = None,  # noqa: F821
                  session: "tensorflow.Session" = None,  # noqa: F821
                  outputs: List[str] = None):
        if graphdef is None:
            assert session is not None
            assert outputs is not None
            graphdef = session.graph_def
            from tensorflow.python.framework import graph_util
            for node in graphdef.node:
                node.device = ""
                graphdef = graph_util.convert_variables_to_constants(
                    session, graphdef, outputs)
        self._graphdef = graphdef
        return self

    @property
    def graphdef(self):
        """
        Returns the wrapped TensorFlow GraphDef.
        """
        return self._graphdef

    def _generate_tree(self) -> dict:
        return {"graphdef": numpy.frombuffer(self._graphdef.SerializeToString(),
                                             dtype=numpy.uint8)}

    def _load_tree(self, tree: dict):
        from tensorflow.core.framework import graph_pb2

        graphdef = graph_pb2.GraphDef()
        graphdef.ParseFromString(tree["graphdef"].data)
        self.construct(graphdef=graphdef)
