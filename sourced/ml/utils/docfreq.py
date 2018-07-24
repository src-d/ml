import logging

from sourced.ml.transformers import BagFeatures2DocFreq, Uast2BagFeatures
from sourced.ml.models import OrderedDocumentFrequencies


def create_or_load_ordered_df(args, ndocs: int=None, bag_features: Uast2BagFeatures=None):
    """
    Returns a preexisting OrderedDocumentFrequencies model from docfreq_in, or generates one
    from the flattened bags of features using args and saves it to docfreq_out.

    :param args: Instance of `argparse.Namespace` that contains docfreq_in, docfreq_out,
                 min_docfreq, and vocabulary_size.
    :param ndocs: Number of documents (can be repos, files or functions)
    :param bag_features: Transformer containing bags of features extracted from the data (the call
                         instantiates an RDD: [(key, doc), val] where key is a specific feature
                         that appeared val times in the document doc.
    :return: OrderedDocumentFrequencies model
    """
    log = logging.getLogger("create_or_load_ordered_df")
    if args.docfreq_in:
        log.info("Loading ordered docfreq model from %s ...", args.docfreq_in)
        return OrderedDocumentFrequencies().load(args.docfreq_in)
    elif ndocs is None or bag_features is None:
        log.error("[IN] only mode, please supply an ordered docfreq model")
        raise ValueError
    log.info("Calculating the document frequencies, hold tight ...")
    df = bag_features \
        .link(BagFeatures2DocFreq()) \
        .execute()
    log.info("Writing ordered docfreq model to %s ...", args.docfreq_out)
    df_model = OrderedDocumentFrequencies() \
        .construct(ndocs, df) \
        .prune(args.min_docfreq) \
        .greatest(args.vocabulary_size) \
        .save(args.docfreq_out)
    return df_model
