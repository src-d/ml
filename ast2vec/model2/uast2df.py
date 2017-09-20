from collections import defaultdict
import logging
import os
import tempfile
from typing import Union

from modelforge import Model

from ast2vec.df import DocumentFrequencies
from ast2vec.uast import UASTModel
from ast2vec.source import Source
from ast2vec.model2.base import Model2Base
from ast2vec.uast_ids_to_bag import UastIds2Bag


class ToDocFreqBase(Model2Base):
    """
    Provides the docfreq state and the function which writes the result.
    It is shared with :class:`Uast2DocFreq` and :class:`MergeDocFreq`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._df = defaultdict(int)
        self._docs = 0

    def finalize(self, index: int, destdir: str):
        model = DocumentFrequencies(log_level=logging.WARNING)
        model.construct(self._docs, self._df.keys(), self._df.values())
        if destdir.endswith(".asdf"):
            path = destdir
        else:
            path = os.path.join(destdir, "docfreq_%d.asdf" % index)
        model.save(path)


class Uast2DocFreq(ToDocFreqBase):
    """
    Calculates document frequencies from models with UASTs.
    """
    MODEL_FROM_CLASS = UASTModel
    MODEL_TO_CLASS = DocumentFrequencies

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._uast2bag = UastIds2Bag(None)

    def convert_model(self, model: Model) -> Union[Model, None]:
        contained = set()
        for uast in model.uasts:
            for key in self._uast2bag.uast_to_bag(uast):
                contained.add(key)
        for word in contained:
            self._df[word] += 1
        self._docs += 1


class MergeDocFreq(ToDocFreqBase):
    """
    Merges several :class:`DocumentFrequencies` models together.
    """
    MODEL_FROM_CLASS = DocumentFrequencies
    MODEL_TO_CLASS = DocumentFrequencies

    def convert_model(self, model: Model) -> Union[Model, None]:
        for word, freq in model:
            self._df[word] += freq
        self._docs += model.docs


def uast2df_entry(args):
    converter = Uast2DocFreq(num_processes=args.processes)
    with tempfile.TemporaryDirectory(dir=args.tmpdir, prefix="source2uast") as tmpdir:
        converter.convert(args.input, tmpdir, pattern=args.filter)
        joiner = MergeDocFreq(num_processes=1)
        joiner.convert(tmpdir, args.output,
                       pattern="%s*.asdf" % DocumentFrequencies.NAME)
