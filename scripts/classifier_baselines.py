"""Run basic classifiers on the train dataset + basic features.
Presupposes that your dataset is in the data/metrics/.

Usage:
    python classifier_baselines.py --dataset <dataset> --clean <clean> --classifier <classifier>"""

import pandas as pd
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report


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
    args = parser.parse_args()

    data = pd.read_csv(
        f"../data/metrics/{args.dataset}_basics.csv",
        dtype={"num_statements": str, "notes": str},
    )

    labels = data["num_statements"]
    train = data.select_dtypes(include=["number"])
    print(
        f"{args.classifier} is being trained using: \n{train.columns} \nParameters: {args}"
    )

    if args.clean == "True":
        train = clean_data(train)

    X_train, X_test, y_train, y_test = train_test_split(
    train, labels, test_size=0.33, random_state=42)

    if args.classifier == "RF":
        clf = RandomForestClassifier(random_state=1, verbose=True).fit(X_train, y_train).predict(X_test)
        print(clf.feature_importances_)
    if args.classifier == "MLP":
        clf = MLPClassifier(random_state=1, max_iter=1000, verbose=True).fit(
            X_train, y_train).predict(X_test)
    if args.classifier == "SVM":
        clf = SVC(
            random_state=1,
            verbose=True,
            max_iter=1000,
        ).fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, zero_division=0))
