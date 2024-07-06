"""Combine test, train and trial data and their features.
Scales everything using the standard scaler by scikit learn. 
Also generates five different splits from the combined data. 

Usage:
    python combine.py
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
    data = data.query('num_statements != 0')
    data = data.drop("num_statements", axis = 1)
    data = data.set_index("sent-id")
    data = data.sort_index()
    return data

if __name__ == "__main__":
    #amr_data = combine_data("amr")
    basics_data = combine_data("basics")
    benepar_data = combine_data("benepar")

    trial = pd.read_csv("../data/trial.csv", usecols = ["sent-id", 'topic', 'phrase', 'phrase_number', 'genre', 'timestamp', 'user',
       'phrase_tokenized', 'num_statements'])
    train = pd.read_csv("../data/train.csv", usecols = ["sent-id", 'topic', 'phrase', 'phrase_number', 'genre', 'timestamp', 'user',
       'phrase_tokenized', 'num_statements'])
    test = pd.read_csv("../data/test.csv", usecols = ["sent-id", 'topic', 'phrase', 'phrase_number', 'genre', 'timestamp', 'user',
       'phrase_tokenized', 'num_statements'])
    datasets = pd.concat([trial, test, train])
    datasets = datasets.query('num_statements != 0')
    datasets.set_index("sent-id", inplace=True)
    datasets = datasets.sort_index()
    combined = pd.concat([basics_data, benepar_data, datasets], axis = 1, sort = True)
    
    scaler = StandardScaler()
    #combined = pd.get_dummies(combined)
    #combined = pd.DataFrame(scaler.fit_transform(combined), columns=combined.columns)
    combined = combined[combined["num_statements"]<=5]

    labels = combined["num_statements"]
    train = combined.drop(columns=["num_statements"])
    labels.to_csv("../data/labels.csv")
    train.to_csv("../data/train_split.csv")

    for i in range(0,5): 
        os.makedirs(f"../results/run_{i}", exist_ok=True)
        X_train, X_test, y_train, y_test = train_test_split(
            train, labels, test_size=0.33, random_state=i, stratify=labels
        )
        X_train.to_csv(f"../results/run_{i}/X_train.csv")
        X_test.to_csv(f"../results/run_{i}/X_test.csv")
        y_train.to_csv(f"../results/run_{i}/y_train.csv")
        y_test.to_csv(f"../results/run_{i}/y_test.csv")