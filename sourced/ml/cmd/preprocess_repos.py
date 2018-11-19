import logging
import os
from uuid import uuid4

from sourced.ml.transformers import create_uast_source, FieldsSelector, Moder, ParquetSaver
from sourced.ml.utils.engine import pause, pipeline_graph


@pause
def preprocess_repos(args):
    log = logging.getLogger("preprocess_repos")
    session_name = "preprocess_repos-%s" % uuid4()

    if os.path.exists(args.output):
        log.critical("%s must not exist", args.output)
        return 1
    if not args.config:
        args.config = []
    root, start_point = create_uast_source(args, session_name)

    start_point \
        .link(Moder(args.mode)) \
        .link(FieldsSelector(fields=args.fields)) \
        .link(ParquetSaver(save_loc=args.output)) \
        .execute()
    pipeline_graph(args, log, root)
