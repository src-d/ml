import glob
import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from tqdm import tqdm

from sourced.ml.cmd.args import handle_input_arg
from sourced.ml.models import Id2Vec


def load_dataset(directory):
    csvpaths = glob.glob(os.path.join(directory, "**/*.csv"))
    chunks = []
    for csvpath in tqdm(csvpaths):
        chunks.append(pd.read_csv(csvpath, sep=',', index_col=None))
    df = pd.concat(chunks)
    df.role = df.role.map(lambda x: x.replace("IDENTIFIER | ", "").replace(" | IDENTIFIER", ""))
    return df


def identifiers_to_datasets(df_unique, id2vecs, log):
    """
    Replace identifiers with its embeddings and create standard X, y dataset
    """
    y = df_unique["role"]
    identifiers = df_unique["identifier"]
    log.info("Final dataset size is %d" % len(identifiers))
    Xs = {}
    for name, id2vec in id2vecs.items():
        X = []
        for key in identifiers:
            X.append(id2vec.embeddings[id2vec[key]])
        Xs[name] = np.array(X)

    return Xs, y


def get_quality(X, y, estimator, tuned_parameters, seed, log):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.33, random_state=seed)

    folding = KFold(n_splits=5, shuffle=True, random_state=seed)
    clf = GridSearchCV(estimator, tuned_parameters, cv=folding, n_jobs=-1)
    clf.fit(X_train, y_train)

    log.debug("Best parameters set found on development set:", clf.best_params_)
    log.debug("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    best_values = (0,)
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        log.debug("\t%0.3f (+/-%0.03f) for %r" % (mean, std, params))
        if best_values[0] < mean:
            best_values = (mean, std, params)
    return clf.score(X_test, y_test), params


def id2role_eval(args):
    """
    Build a simple log-regression model to predict a Role of UAST node by its identifier embedding.
    It creates a report about embedding quality.
    To collect the dataset please use repos2roleids entry point.
    """
    log = logging.getLogger("id2role_eval")

    models = {}
    common_tokens = None
    for path in handle_input_arg(args.models, log):
        name = os.path.split(path)[1]
        id2vec = Id2Vec().load(path)
        id2vec.construct(id2vec.embeddings, [t.split(".")[-1] for t in id2vec.tokens])
        models[name] = id2vec
        if common_tokens is None:
            common_tokens = set(models[name].tokens)
        else:
            common_tokens &= set(models[name].tokens)
    log.info("Common tokens in all models: %d" % len(common_tokens))

    tuned_parameters = [{'C': [10 ** x for x in range(-7, -1)]}]
    # Load data and preprocess
    log.info("Data loading...")
    df = load_dataset(args.dataset)
    df_ids = set(df["identifier"])
    valid_tokens = list(set(df_ids) & common_tokens)
    df = df[df["identifier"].isin(valid_tokens)]
    # Count identifiers in dataset
    log.info("Have embeddings only for %d tokens from %d in your dataset" % (
        len(valid_tokens), len(df_ids)))
    df_unique = df.groupby("identifier").agg(lambda x: x.value_counts().index[0])
    df_unique["identifier"] = df_unique.index
    # Exclude rare roles
    vc = df["role"].value_counts()
    del df
    rare = set(vc[vc < 10].index)
    log.info("%d rare roles excluded. " % len(rare))
    df_unique = df_unique[~df_unique["role"].isin(rare)]
    log.debug("Convert words to its embeddings")
    Xs, y = identifiers_to_datasets(df_unique, models, log)

    final_report = pd.DataFrame(columns=["embedding name", "score", "best C value"])
    for name in tqdm(Xs):
        log.info("{}...".format(name))
        best_values = get_quality(Xs[name], y,
                                  LogisticRegression(class_weight="balanced",
                                                     random_state=args.seed),
                                  tuned_parameters=tuned_parameters,
                                  seed=args.seed,
                                  log=log)
        final_report = final_report.append({"embedding name": name,
                                            "score": best_values[0],
                                            "best C value": best_values[1]["C"]},
                                           ignore_index=True)

    print("Pairs number: %d.\n" % len(valid_tokens))
    print(final_report)
