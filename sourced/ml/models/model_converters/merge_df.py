from collections import defaultdict
import os

from sourced.ml.models.model_converters.base import Model2Base
from sourced.ml.models.df import DocumentFrequencies
from sourced.ml.models.ordered_df import OrderedDocumentFrequencies


class MergeDocFreq(Model2Base):
    """
    Merges several :class:`DocumentFrequencies` models together.
    """
    MODEL_FROM_CLASS = DocumentFrequencies
    MODEL_TO_CLASS = DocumentFrequencies

    def __init__(self, min_docfreq: int, vocabulary_size: int, ordered: bool=False,
                 *args, **kwargs):
        super().__init__(num_processes=1, *args, **kwargs)
        self.ordered = ordered
        self.min_docfreq = min_docfreq
        self.vocabulary_size = vocabulary_size
        self._df = defaultdict(int)
        self._docs = 0

    def convert_model(self, model: DocumentFrequencies) -> None:
        for word, freq in model:
            self._df[word] += freq
        self._docs += model.docs

    def finalize(self, index: int, destdir: str):
        df_model = OrderedDocumentFrequencies if self.ordered else DocumentFrequencies
        df_model(log_level=self._log.level) \
            .construct(self._docs, self._df) \
            .prune(self.min_docfreq) \
            .greatest(self.vocabulary_size) \
            .save(self._save_path(index, destdir))

    @staticmethod
    def _save_path(index: int, destdir: str):
        if destdir.endswith(".asdf"):
            return destdir
        return os.path.join(destdir, "docfreq_%d.asdf" % index)
