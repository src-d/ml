import logging
import string
import tarfile
import warnings

import numpy
from keras.preprocessing.sequence import pad_sequences
from modelforge.progress_bar import progress_bar


DEFAULT_MAX_IDENTIFIER_LEN = 40  # default max length of sequence
PADDING = "post"  # add padding values after input
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SHUFFLE_VALUE = True

# In the CSV file, columns 0,1,2 contain statistics about the identifier.
CSV_IDENTIFIERS_COL = 3  # Column 3 contains the input identifier e.g. "FooBar".
CSV_SPLIT_IDENTIFIERS_COL = 4  # Column 4 contains the identifier lowercase and spitted "foo bar".


def read_identifiers(csv_path: str, use_header: bool=True,
                     max_identifier_len: int=DEFAULT_MAX_IDENTIFIER_LEN,
                     identifiers_col: int=CSV_IDENTIFIERS_COL,
                     split_identifiers_col: int=CSV_SPLIT_IDENTIFIERS_COL,
                     shuffle: bool=True):
    """
    Reads and filters too long identifiers from CSV file.
    :param csv_path: path to the CSV file.
    :param use_header: uses header as normal line (True) or treat as header line with column names.
    :param max_identifier_len: maximum length of raw identifier. Skip identifier if longer.
    :param identifiers_col: column in CSV file for the raw identifier.
    :param split_identifiers_col: column in CSV file for the splitted identifier.
    :param shuffle: indicates whether to reorder the list of identifiers
        at random after reading it.
    :return: list of splitted identifiers.
    """
    log = logging.getLogger("read_identifiers")
    log.info("Reading data from the CSV file %s", csv_path)
    identifiers = []
    # TODO: Update dataset loading as soon as https://github.com/src-d/backlog/issues/1212 done.
    # Think about dataset download step.
    with tarfile.open(csv_path, encoding="utf-8") as f:
        assert len(f.members) == 1, "One archived file is expected, got: %s" % len(f.members)
        content = f.extractfile(f.members[0])
        if not use_header:
            content.readline()
        for line in progress_bar(content.readlines(), log):
            row = line.decode("utf-8").strip().split(",")
            if len(row[identifiers_col]) <= max_identifier_len:
                identifiers.append(row[split_identifiers_col])
    if shuffle:
        numpy.random.shuffle(identifiers)
    log.info("Number of identifiers after filtering: %s." % len(identifiers))
    return identifiers


def prepare_features(csv_path: str, use_header: bool=True,
                     max_identifier_len: int=DEFAULT_MAX_IDENTIFIER_LEN,
                     identifiers_col: int=CSV_IDENTIFIERS_COL,
                     split_identifiers_col: int=CSV_SPLIT_IDENTIFIERS_COL,
                     shuffle: bool=DEFAULT_SHUFFLE_VALUE, test_size: float=DEFAULT_TEST_SIZE,
                     padding: str=PADDING):
    """
    Prepare the features for training the identifier splitting task.
    :param csv_path: path to the CSV file.
    :param use_header: uses header as normal line (True) or treat as header line with column names.
    :param max_identifier_len: maximum length of raw identifier. Skip identifier if longer.
    :param identifiers_col: column in CSV file for the raw identifier.
    :param split_identifiers_col: column in CSV file for the splitted identifier.
    :param shuffle: indicates whether to reorder the list of identifiers at random after reading it
    :param test_size: Proportion of test samples used for evaluation.
    :param padding: position where to add padding values:
        after the intput sequence if "post", before if "pre".
    :return: training and testing features to train the neural net for the splitting task.
    """
    log = logging.getLogger("prepare_features")

    # read data from file
    identifiers = read_identifiers(csv_path=csv_path, use_header=use_header,
                                   max_identifier_len=max_identifier_len,
                                   identifiers_col=identifiers_col,
                                   split_identifiers_col=split_identifiers_col, shuffle=shuffle)

    # convert identifiers into character indices and labels
    log.info("Converting identifiers to character indices")
    log.info("Number of identifiers: %d, Average length: %d characters" %
             (len(identifiers), numpy.mean([len(i) for i in identifiers])))

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
            elif char == " ":
                split_arr.append(1)
                skip_char = True
            else:
                log.warning("Unexpected symbol %s in identifier", char)
        # sanity check
        assert len(index_arr) == len(split_arr)
        char_id_seq.append(index_arr)
        splits.append(split_arr)

    log.info("Number of subtokens: %d, Number of distinct characters: %d" %
             (sum([sum(split_arr) for split_arr in splits]) + len(identifiers),
              len(set([i for index_arr in char_id_seq for i in index_arr]))))

    # train/test splitting
    log.info("Train/test splitting...")
    n_train = int((1 - test_size) * len(char_id_seq))
    x_train = char_id_seq[:n_train]
    x_test = char_id_seq[n_train:]
    y_train = splits[:n_train]
    y_test = splits[n_train:]
    log.info("Number of train samples: %s, number of test samples: %s" % (len(x_train),
                                                                          len(x_test)))

    # pad sequence
    log.info("Padding of the sequences...")
    x_train = pad_sequences(x_train, maxlen=max_identifier_len, padding=padding)
    x_test = pad_sequences(x_test, maxlen=max_identifier_len, padding=padding)
    y_train = pad_sequences(y_train, maxlen=max_identifier_len, padding=padding)
    y_test = pad_sequences(y_test, maxlen=max_identifier_len, padding=padding)

    return x_train, x_test, y_train[:, :, None], y_test[:, :, None]
