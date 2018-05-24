import logging
import string
import tarfile
import warnings

import numpy as np
try:
    from keras.preprocessing.sequence import pad_sequences
except ImportError as e:
    warnings.warn("Tensorflow or/and Keras are not installed, dependent functionality is "
                  "unavailable.")


MAXLEN = 40  # max length of sequence
PADDING = "post"  # add padding values after input

# CSV default parameters
TOKEN_COL = 3
TOKEN_SPLIT_COL = 4


def read_identifiers(csv_loc, use_header=True, mode="r", maxlen=MAXLEN, token_col=TOKEN_COL,
                     token_split_col=TOKEN_SPLIT_COL, shuffle=True):
    """
    Read and filter too long identifiers from CSV file.
    :param csv_loc: location of CSV file
    :param use_header: use header as normal line (True) or treat as header line with column names
    :param mode: mode to read tarfile
    :param maxlen: maximum length of raw identifier. If it's longer than skip it
    :param token_col: column in CSV file for raw token
    :param token_split_col: column in CSV file for splitted token
    :param shuffle: shuffle or not list of identifiers
    :return: list of splitted tokens
    """
    log = logging.getLogger("id-splitter-prep")
    log.info("Reading data from CSV...")
    identifiers = []
    with tarfile.open(csv_loc, mode=mode, encoding="utf-8") as f:
        assert len(f.members) == 1, "Expect one archived file"
        content = f.extractfile(f.members[0])
        if not use_header:
            content.readline()
        for line in content:
            parts = line.decode("utf-8").strip().split(",")
            if len(parts[token_col]) <= maxlen:
                identifiers.append(parts[token_split_col])
    if shuffle:
        np.random.shuffle(identifiers)
    log.info("Number of identifiers after filtering: {}.".format(len(identifiers)))
    return identifiers


def prepare_features(csv_loc, use_header=True, token_col=TOKEN_COL, maxlen=MAXLEN, mode="r",
                     token_split_col=TOKEN_SPLIT_COL, shuffle=True, test_size=0.2,
                     padding=PADDING):
    log = logging.getLogger("id-splitter-prep")

    # read data from file
    identifiers = read_identifiers(csv_loc=csv_loc, use_header=use_header, token_col=token_col,
                                   maxlen=maxlen, mode=mode, token_split_col=token_split_col,
                                   shuffle=shuffle)

    # convert identifiers into character indices and labels
    log.info("Converting identifiers to character indices...")

    char2ind = dict((c, i + 1) for i, c in enumerate(sorted(string.ascii_lowercase)))

    char_id_seq = []
    splits = []
    for identifier in identifiers:
        # iterate through identifier and convert to array of char indices & boolean split array
        index_arr = []
        split_arr = []
        skip_char = False
        for char in identifier.strip():
            if char in char2ind:
                index_arr.append(char2ind[char])
                if skip_char:
                    skip_char = False
                    continue
                split_arr.append(0)
            else:
                # space
                split_arr.append(1)
                skip_char = True
        # sanity check
        assert len(index_arr) == len(split_arr)
        char_id_seq.append(index_arr)
        splits.append(split_arr)

    # train/test splitting
    log.info("Train/test splitting...")
    n_train = int((1 - test_size) * len(char_id_seq))
    x_tr = char_id_seq[:n_train]
    x_t = char_id_seq[n_train:]
    y_tr = splits[:n_train]
    y_t = splits[n_train:]
    log.info("Number of train samples: {}, number of test samples: {}.".format(len(x_tr),
                                                                               len(x_t)))

    # pad sequence
    log.info("Padding of the sequences...")
    x_tr = pad_sequences(x_tr, maxlen=maxlen, padding=padding)
    x_t = pad_sequences(x_t, maxlen=maxlen, padding=padding)
    y_tr = pad_sequences(y_tr, maxlen=maxlen, padding=padding)
    y_t = pad_sequences(y_t, maxlen=maxlen, padding=padding)

    return x_tr, x_t, y_tr[:, :, None], y_t[:, :, None]
