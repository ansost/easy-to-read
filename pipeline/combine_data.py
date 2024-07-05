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

def read_in(set:str)-> pd.DataFrame:
    dataset = pd.read_csv(f"../data/{set}.csv")
    benepar = pd.read_csv(f"../data/metrics/benepar_features_{set}.csv", usecols=["is_sent", "big_np_count", "big_pp_count"])
    basics = pd.read_csv(f"../data/metrics/{set}_basics.csv", usecols=[
        "disj_count",
        "mean_dep_length",
        "conj_count",
        "max_dep_length",
        "token_count",
        "verb_count",
        "min_dep_length",
        "flesh_reading_ease",
        "class_score",
    ])
    if set == "trial":
        combined = pd.concat([dataset, benepar, basics], axis = 1) # 1 for columns
    else: 
        amr = pd.read_csv(f"../data/metrics/{set}_amr.csv", usecols=["amr", "amr_prolog"])
        combined = pd.concat([dataset, benepar, amr, basics], axis = 1) # 1 for columns
    return combined

if __name__ == "__main__":
    test = read_in("test")
    train = read_in("train")
    trial  = read_in("trial")
    combined = pd.concat([train, test, trial], axis = 0) # 0 for index

    scaler = StandardScaler()
    combined = pd.get_dummies(combined)
    combined = pd.DataFrame(scaler.fit_transform(combined), columns=combined.columns)
    combined = combined[combined["num_statements"] != "0"]
    combined = combined['num_statements'].apply(lambda x: len(x) > 3) 

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