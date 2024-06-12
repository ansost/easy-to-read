"""Run basic classifiers on the train dataset + basic features.
Presupposes that your dataset is in the data/metrics/.

Usage:
    python classifier_baselines.py --dataset <dataset> --clean <clean> --classifier <classifier> --zero_class <zero_class>"""

import pandas as pd
import argparse

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from embeddings import *
from classifiers import *


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
        required=False,
        help="'True' or 'False'.Whether to remove the columns present in the original train dataset, like the tokenized phrase, notes, etc.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="Which classifier to use. Options: Random Forrst 'RF', Multi-layer Perceptron 'MLP', Support Vector Machine 'SVM', or logistic regression 'regression'.",
    )
    parser.add_argument(
        "--zero_class",
        type=str,
        default="keep",
        required=False,
        help="Whether to keep or remove the 0 labels ('keep', 'remove'). Default: keep.",
    )
    args = parser.parse_args()

    assert args.classifier in [
        "RF",
        "MLP",
        "SVM",
        "regression",
    ], "Invalid classifier (RF, MLP, SVM, regression are available)."

    data = pd.read_csv(
        f"../data/metrics/{args.dataset}_basics.csv",
        dtype={"num_statements": str, "notes": str},
    )
    data = data.groupby("num_statements").filter(lambda x: len(x) > 3)

    if args.zero_class == "remove":
        train = data[data["num_statements"] != "0"]
        labels = train["num_statements"]
    else:
        train = data
        labels = data["num_statements"]
    train = train.select_dtypes(include=["number"])

    scaler = StandardScaler()
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)

    print(f"Results from {args.classifier.upper()} using: {list(train.columns)}")
    if args.clean == "True":
        train.drop(columns=["sent-id"], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        train, labels, test_size=0.33, random_state=42, stratify=labels
    )

    preds, weights = eval(args.classifier)(X_train, X_test, y_train)

    print("Classification score:\n")
    print(weights)
    print(classification_report(y_test, preds, zero_division=0))
