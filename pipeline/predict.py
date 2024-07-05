"""Generate predictions in submission format.

Usage:
    python 
"""
import pandas as pd 
from sklearn.metrics import classification_report

from src.ml import *
from src.parser import *

def load_data(run):
    X_train = pd.read_csv(f"../results/run_{run}/X_train.csv")
    X_test = pd.read_csv(f"../results/run_{run}/X_test.csv")
    y_train = pd.read_csv(f"../results/run_{run}/y_train.csv")
    y_test = pd.read_csv(f"../results/run_{run}/y_test.csv")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    classifiers = {"RF":"RF", "MLP":"MLP", "SVM":"SVM", "regression": "regression"}

    for run in ["0","1","2","3","4"]:
        X_train, X_test, y_train, y_test = load_data(run)
        for classifier in classifiers.keys():
            predictions = classifiers[classifier](X_train, X_test, y_train)
            accuracy = classification_report(y_test["num_statements"], predictions, zero_division=0, output_dict=True)
            print(classifier, accuracy)

            submission = pd.DataFrame()
            submission["sent-id"] = y_test["sent-id"]
            submission["num_statements"] = predictions
            submission["statement_spans"] = ""
            submission.to_csv(f"../results/{classifier}_submission.csv")