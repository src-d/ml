import os
from scipy.sparse import vstack

from sourced.ml import extractors
from sourced.ml.models.bow import BOW
from sourced.ml.models.model_converters.base import Model2Base


class MergeBOW(Model2Base):
    """
    Merges several :class:`BOW` models together.
    """
    MODEL_FROM_CLASS = BOW
    MODEL_TO_CLASS = BOW

    def __init__(self, features=None, *args, **kwargs):
        super().__init__(num_processes=1, *args, **kwargs)
        self.documents = None
        self.tokens = None
        self.matrix = None
        self.deps = None
        self.features_namespaces = None
        if features:
            self.features_namespaces = [ex.NAMESPACE for ex in extractors.__extractors__.values()
                                        if ex.NAME in features]

    def convert_model(self, model: BOW) -> None:
        if self.tokens is None:
            self.tokens = model.tokens
            self.documents = model.documents
            self.matrix = [model.matrix.tocsr()]
            self.deps = model._meta["dependencies"]
        elif set(self.tokens) != set(model.tokens):
            raise ValueError("Models don't share the same set of tokens !")
        else:
            self.documents += model.documents
            self.matrix.append(model.matrix.tocsr())

    def finalize(self, index: int, destdir: str):
        self._log.info("Stacking matrices ...")
        matrix = self.matrix.pop(0)
        while self.matrix:
            matrix = vstack([matrix, self.matrix.pop(0)])
            self._log.info("%s matrices to stack ...", len(self.matrix))
        self.matrix = matrix
        self._log.info("Writing model ...")
        if self.features_namespaces:
            self._reduce_matrix()
        BOW(log_level=self._log.level) \
            .construct(self.documents, self.tokens, self.matrix) \
            .save(self._save_path(index, destdir), self.deps)

    def _reduce_matrix(self):
        reduced_tokens = []
        columns = []
        matrix = self.matrix.tocsc()
        for i, token in enumerate(self.tokens):
            if token.split(".")[0] in self.features_namespaces:
                reduced_tokens.append(token)
                columns.append(i)
        self.tokens = reduced_tokens
        self.matrix = matrix[:, columns]

    @staticmethod
    def _save_path(index: int, destdir: str):
        if destdir.endswith(".asdf"):
            return destdir
        return os.path.join(destdir, "bow_%d.asdf" % index)
