"""Generate predictions in submission format.

Usage:
    python predict.py
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report

from src.ml import *
from src.parser import *


def load_data(run):
    X_train = pd.read_csv(f"../results/run_{run}/X_train.csv").drop(
        ["topic", "phrase_number", "genre", "timestamp", "user", "phrase_tokenized"],
        axis=1,
    )
    X_test = pd.read_csv(f"../results/run_{run}/X_test.csv").drop(
        ["topic", "phrase_number", "genre", "timestamp", "user", "phrase_tokenized"],
        axis=1,
    )
    y_train = pd.read_csv(f"../results/run_{run}/y_train.csv")
    y_test = pd.read_csv(f"../results/run_{run}/y_test.csv")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        default="eval",
        choices=["eval", "submission"],
        help="Wether to do testing with all classifiers or generate a submission.",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=False,
        default="parser",
        choices=["RF", "MLP", "SVM", "regression", "parser"],
        help="Specific source for predictions to use for submission.",
    )
    args = parser.parse_args()
    classifiers = {"RF": RF, "MLP": MLP, "SVM": SVM, "regression": regression}

    if args.mode == "eval":

        for run in ["0", "1", "2", "3", "4"]:
            X_train, X_test, y_train, y_test = load_data(run)
            df = pd.DataFrame()
            df["sent-id"] = y_test["sent-id"]
            df["gold_label"] = y_test["num_statements"]

            for classifier in classifiers.keys():
                predictions = classifiers[classifier](
                    X_train.drop("phrase", axis=1),
                    X_test.drop("phrase", axis=1),
                    y_train.drop("sent-id", axis = 1).values.ravel(),
                )
                temp = pd.DataFrame(predictions, columns = [classifier])
                df = pd.concat([temp, df], axis =1)

            #df["parser"] = X_test["phrase"].apply(count_statements)
            df.to_csv(f"../results/run_{run}/all_predictions.csv")

    if args.mode == "submission":
        submission = pd.DataFrame()
        submission["sent-id"] = y_test["sent-id"]
        submission["statement_spans"] = ""

        eval_data = pd.read_csv("../data/eval.csv")

        if args.classifier == "parser":
            submission["num_statements"] = X_test["phrase"].apply(count_statements)
        else:
            predictions = classifiers[classifier](X_train.drop("phrase", axis=1), X_test.drop("phrase", axis=1), y_train.drop("sent-id", axis = 1).values.ravel())
            df = pd.DataFrame(predictions, columns = ["sent-id", "num_statements"])

        submission.to_csv(f"../results/{classifier}_submission.csv")
