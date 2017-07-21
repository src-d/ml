import modelforge.dump


def dump_model(args):
    """
    Proxy for :func:`modelforge.dump.dump_model`.
    """
    if args.gcs:
        args.args = "bucket=" + args.gcs
        del args.gcs
    else:
        args.args = ""
    args.backend = None
    return modelforge.dump.dump_model(args)
