import os
import logging
import argparse
import tempfile
import unittest

from sourced.ml.utils.docfreq import create_or_load_ordered_df
from sourced.ml.utils import create_spark
from sourced.ml.transformers import ParquetLoader, UastDeserializer, UastRow2Document, Counter, \
    Uast2BagFeatures
from sourced.ml.extractors import IdentifiersBagExtractor

import sourced.ml.tests.models as paths


class DocumentFrequenciesUtilTests(unittest.TestCase):

    def test_load(self):
        args = argparse.Namespace(docfreq_in=paths.DOCFREQ, docfreq_out=None, min_docfreq=None,
                                  vocabulary_size=None)
        df_model = create_or_load_ordered_df(args, None, None)
        self.assertEqual(df_model.docs, 1000)

    def test_create(self):
        session = create_spark("test_df_util")
        uast_extractor = ParquetLoader(session, paths.PARQUET_DIR) \
            .link(UastRow2Document())
        ndocs = uast_extractor.link(Counter()).execute()
        uast_extractor = uast_extractor.link(UastDeserializer()) \
            .link(Uast2BagFeatures([IdentifiersBagExtractor()]))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "df.asdf")
            args = argparse.Namespace(docfreq_in=None, docfreq_out=tmp_path, min_docfreq=1,
                                      vocabulary_size=1000)
            df_model = create_or_load_ordered_df(args, ndocs, uast_extractor)
            self.assertEqual(df_model.docs, ndocs)
            self.assertTrue(os.path.exists(tmp_path))
