import os
import logging
from typing import List

from sourced.ml.models import QuantizationLevels
from sourced.ml.transformers import Uast2Quant
from sourced.ml.extractors import BagsExtractor


def create_or_apply_quant(model_path: str, extractors: List[BagsExtractor], extracted_uasts=None):
    log = logging.getLogger("create_or_apply_quant")
    if os.path.exists(model_path):
        log.info("Loading the quantization levels from %s and applying quantization to supported"
                 " extractors...", model_path)
        try:
            QuantizationLevels().load(source=model_path).apply_quantization(extractors)
        except (ValueError, ImportError):
            pass
        else:
            return
    if extracted_uasts is None:
        log.error("[IN] only mode, please supply a quantization levels model")
        raise ValueError
    else:
        quant = Uast2Quant(extractors)
        extracted_uasts.link(quant).execute()
        if quant.levels:
            log.info("Writing quantization levels to %s", model_path)
            QuantizationLevels().construct(quant.levels).save(model_path)
