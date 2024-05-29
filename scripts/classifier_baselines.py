"""Run basic classifiers on the train dataset + basic features.
Presupposes that your dataset is in the data/metrics/.

Usage:
    python classifier_baselines.py --dataset <dataset> --clean <clean> --classifier <classifier>"""

import pandas as pd
import argparse
from transformers import pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score

from . import embeddings
from embeddings import *


def get_embeddings(phrase: str):
    return pipe(phrase)[0]["token_str"]


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove original train dataset columns from df."""
    out = [
        "sent-id",
        "topic",
        "phrase",
        "phrase_number",
        "genre",
        "timestamp",
        "user",
        "phrase_tokenized",
        "statement_spans",
        "notes",
        "pos_onehot",
    ]
    cols = data.columns
    for name in out:
        if name in cols:
            data.drop(columns=[name], inplace=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Any .csv file in data/metrics/, that includes a column 'num_statements'.",
    )
    parser.add_argument(
        "--clean",
        type=bool,
        required=True,
        help="'True' or 'False'.Whether to remove the columns present in the original train dataset, like the tokenized phrase, notes, etc.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="Which classifier to use. Options: Random Forrst 'RF', Multi-layer Perceptron 'MLP'",
    )
    parser.add_argument(
        "--add_embeddings",
        type=str,
        required=True,
        default="True",
        help="Whether to add embeddings to the df. Default: True.",
    )
    args = parser.parse_args()

    classifiers = {
        "RF": RandomForestClassifier,
        "MLP": MLPClassifier,
        "SVM": SVC,
    }

    assert args.classifier in [
        "RF",
        "MLP",
        "SVM",
    ], "Invalid classifier (RF, MLP, SVM are available)."

    data = pd.read_csv(
        f"../data/metrics/{args.dataset}_basics.csv",
        dtype={"num_statements": str, "notes": str},
    )

    if args.embeddings == "True":
        if "embeddings" not in data.columns:
            pipe = pipeline("fill-mask", model="google-bert/bert-base-german-cased")
            data["embeddings"] = data["phrase"].apply(get_embeddings)
            data.to_csv(f"../data/metrics/{args.dataset}.csv", index=False)

    data = data.groupby("num_statements").filter(lambda x: len(x) > 1)

    labels = data["num_statements"]
    train = data.select_dtypes(include=["number"])
    print(
        f"{args.classifier} is being trained using: \n{train.columns} \nParameters: {args}"
    )

    if args.clean == "True":
        train = clean_data(train)

    X_train, X_test, y_train, y_test = train_test_split(
        train, labels, test_size=0.33, random_state=42, stratify=labels
    )

    clf = classifiers[args.classifier](random_state=1).fit(X_train, y_train)
    preds = clf.predict(X_test)

    if args.classifier == "RF":
        print(clf.feature_importances_)

    print("Classification score:\n")
    print(classification_report(y_test, preds, zero_division=0))
    print("Crossvalidation score:\n")
    print(cross_val_score(clf, train, labels, cv=3, scoring="accuracy"))
