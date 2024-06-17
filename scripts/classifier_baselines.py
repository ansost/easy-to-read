"""Run basic classifiers on the train dataset + basic features.
Presupposes that your dataset is in the data/metrics/.

Usage:
    python classifier_baselines.py --dataset <dataset> --clean <clean> --classifier <classifier> --zero_class <zero_class>"""

import pandas as pd
import argparse

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classifiers import *


def load_data():
    """Load and filter data based on the arguments given."""

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["train", "test", "trial"],
        help="Which dataset to use for the classifier. Options: 'train', 'dev', 'test'.",
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

    data = pd.read_csv(
        f"../data/{args.dataset}.csv",
        dtype={"num_statements": str, "notes": str},
    )

    # Load extra features.
    if not args.use_benepar and not args.use_basics and not args.use_amr:
        raise ValueError("No features selected. Please select one or more features.")

    if args.use_benepar:
        benepar_df = pd.read_csv(
            f"../data/metrics/benepar_features_{args.dataset}.csv",
            usecols=["is_sent", "big_np_count", "big_pp_count"],
        )
        data = pd.concat([data, benepar_df], axis=1)
        del benepar_df
    if args.use_basics:
        basics_df = pd.read_csv(
            f"../data/metrics/{args.dataset}_basics.csv",
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
        data = pd.concat([data, basics_df], axis=1)
        del basics_df
    if args.use_amr:
        amr_df = pd.read_csv(
            f"../data/metrics/{args.dataset}_amr.csv",
            usecols=["amr", "amr_prolog"],
        )
        data = pd.concat([data, amr_df], axis=1)
        del amr_df

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
        f"../results/classification_reports/{args.dataset}_{args.classifier}_{args.boundary_classes}.tex",
        index=False,
        float_format="%.2f",
    )
