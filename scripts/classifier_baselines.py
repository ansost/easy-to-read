"""Run basic classifiers on the train dataset + basic features.
Presupposes that your dataset is in the data/metrics/.

Usage:
    python classifier_baselines.py --dataset <str> --classifier <str> --boundary_classes <str> --use_benepar <bool> --use_basics <bool> --use_amr <bool>
"""

import pandas as pd
import argparse

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classifiers import *


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
        ],
    )


def read_benepar(dataset):
    return pd.read_csv(
        f"../data/metrics/benepar_features_{dataset}.csv",
        usecols=["is_sent", "big_np_count", "big_pp_count"],
    )


def read_amr(dataset):
    return pd.read_csv(
        f"../data/metrics/{dataset}_amr.csv", usecols=["amr", "amr_prolog"]
    )


def combine_data(dataset, feature):
    if dataset == "combined":
        train = globals()[f"read_{feature}"]("train")
        test = globals()[f"read_{feature}"]("test")
        data = pd.concat([train, test], ignore_index=True, axis=1)
    else:
        data = globals()[f"read_{feature}"](dataset)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["train", "test", "trial", "combined"],
        help="Which dataset to use for the classifier. Options: 'train', 'dev', 'test'. The combination 'trial' with 'use_amr' is not available.",
    )
    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        required=True,
        choices=["RF", "MLP", "SVM", "regression"],
        help="Which classifier to use. Options: Random Forrst 'RF', Multi-layer Perceptron 'MLP', Support Vector Machine 'SVM', or logistic regression 'regression'.",
    )
    parser.add_argument(
        "-b",
        "--boundary_classes",
        type=str,
        required=False,
        choices=["remove_0", "remove_6", "remove_0_6", "misc_class"],
        help="How to handle classes with label 0, and 6 and above. For more information see the paper. Options: 'remove_0', 'remove_6', 'remove_0_6', 'misc_class'.",
    )
    parser.add_argument(
        "-bene",
        "--use_benepar",
        type=bool,
        default=False,
        required=False,
        help="Whether to include benepar features.",
    )
    parser.add_argument(
        "-basics",
        "--use_basics",
        type=bool,
        default=False,
        required=False,
        help="Whether to include basic features.",
    )
    parser.add_argument(
        "-amr",
        "--use_amr",
        type=bool,
        default=False,
        required=False,
        help="Whether to include AMR features.",
    )
    args = parser.parse_args()
    datasets = ["train", "test"]

    if args.dataset == "combined":
        data = pd.concat(
            [
                pd.read_csv(
                    f"../data/train.csv",
                    dtype={"num_statements": str, "phrase": str},
                    usecols=["num_statements", "phrase"],
                ),
                pd.read_csv(
                    f"../data/test.csv",
                    dtype={"num_statements": str, "phrase": str},
                    usecols=["num_statements", "phrase"],
                ),
            ],
            ignore_index=True,
        )
    else:
        data = pd.read_csv(
            f"../data/{args.dataset}.csv",
            dtype={"num_statements": str, "phrase": str},
            usecols=["num_statements", "phrase"],
        )

    # Load extra features.
    if not args.use_benepar and not args.use_basics and not args.use_amr:
        raise ValueError("No features selected. Please select one or more features.")

    if args.use_benepar:
        data = combine_data(args.dataset, "benepar")
    if args.use_basics:
        data = combine_data(args.dataset, "basics")
    if args.use_amr:
        data = combine_data(args.dataset, "amr")

    breakpoint()
    # Exclude or re-group boundary classes.
    data = data.groupby("num_statements").filter(lambda x: len(x) > 3)
    if args.boundary_classes:
        if args.boundary_classes == "remove":
            data = data[data["num_statements"] != "0"]
        elif args.boundary_classes == "remove_6":
            data = data[data["num_statements"] < "6"]
        elif args.boundary_classes == "remove_0_6":
            data = data[data["num_statements"] != "0"]
            data = data[data["num_statements"] != "6"]
        elif args.boundary_classes == "misc_class":
            data["num_statements"] = data["num_statements"].apply(
                lambda x: x if x != "0" and x < "6" else "misc"
            )

    labels = data["num_statements"]
    train = data.drop(columns=["num_statements"])
    del data

    train = pd.get_dummies(train)
    scaler = StandardScaler()
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        train, labels, test_size=0.33, random_state=42, stratify=labels
    )

    preds, weights = eval(args.classifier)(X_train, X_test, y_train)
    res = classification_report(y_test, preds, zero_division=0, output_dict=True)
    report = pd.DataFrame(res).transpose().round(2)
    report.to_csv(
        f"../results/classification_reports/{args.dataset}_{args.classifier}_{args.boundary_classes}.csv"
    )
    report.to_latex(
        f"../results/classification_reports/tex_tables/{args.dataset}_{args.classifier}_{args.boundary_classes}.tex",
        index=False,
        float_format="%.2f",
    )
