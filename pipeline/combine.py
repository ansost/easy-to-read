"""Combine test, train and trial data and their features.
Scales everything using the standard scaler by scikit learn. 
Also generates five different splits from the combined data. 

Usage:
    python combine_data.py
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def read_basics(dataset):
    return pd.read_csv(
        f"../data/metrics/{dataset}_basics.csv",
        usecols=[
            "disj_count",
            "mean_dep_length",
            "conj_count",
            "max_dep_length",
            "token_count",
            "verb_count",
            "min_dep_length",
            "flesh_reading_ease",
            "class_score",
            "sent-id",
            "num_statements"
        ],
    )


def read_benepar(dataset):
    return pd.read_csv(
        f"../data/metrics/benepar_features_{dataset}.csv",
        usecols=["is_sent", "big_np_count", "big_pp_count", "sent-id", "num_statements"],
    )


def read_amr(dataset):
    return pd.read_csv(
        f"../data/metrics/{dataset}_amr.csv", usecols=["amr", "amr_prolog", "sent-id", "num_statements"]
    )

def combine_data(feature):
    train = globals()[f"read_{feature}"]("train")
    test = globals()[f"read_{feature}"]("test")
    if feature != "amr":
        trial = globals()[f"read_{feature}"]("trial")
        data = pd.concat([train, test, trial])
    else:
        data = pd.concat([train, test], axis = 0)
    amr_data = amr_data.query('num_statements != 0')
    return data

if __name__ == "__main__":
    amr_data = combine_data("amr")
    basics_data = combine_data("basics")
    benepar_data = combine_data("benepar")

    trial = pd.read_csv("../data/trial.csv")
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    datasets = pd.concat(["trial", "test", "train"])
    datasets = datasets.query('num_statements != 0')

    assert len(amr_data) == len(basics_data) == len(benepar_data) == len(datasets)
    combined = pd.concat([amr_data, basics_data, benepar_data, datasets], axis = 1)
    
    scaler = StandardScaler()
    combined = pd.get_dummies(combined)
    combined = pd.DataFrame(scaler.fit_transform(combined), columns=combined.columns)

    labels = combined[["num_statements", "sent-id"]]
    train = combined.drop(columns=["num_statements"])

    for i in range(0,5):
        os.makedirs("../results/run_{i}", exist_ok=True)
        X_train, X_test, y_train, y_test = train_test_split(
            train, labels, test_size=0.33, random_state=42, stratify=labels
        )
        X_train.to_csv("../results/run_{i}/X_train.csv")
        X_test.to_csv("../results/run_{i}/X_test.csv")
        y_train.to_csv("../results/run_{i}/y_train.csv")
        y_test.to_csv("../results/run_{i}/y_test.csv")