import numpy


def log_tf_log_idf(tf, df, ndocs):
    return numpy.log(1 + tf) * numpy.log(ndocs / df)
