"""Compare num_statement predictions from AMR and rule-based approaches

Prequisites:
    - results from AME based approach are available
    - results from rule based approach are available

Usage:
    python check_errors.py
"""

import pandas as pd
import json


def get_error_df(predictions, sent_ids, y_gold):
    error_df = pd.DataFrame(
        columns=[
            "sent-id",
            "regression",
            "SVM",
            "MLP",
            "RF",
            "all_true",
            "all_false",
            "one_true",
        ],
        index=range(len(predictions)),
    )

    for i in range(len(predictions)):
        error_df.loc[i, "sent-id"] = sent_ids[i]

        error_df.loc[i, "regression"] = predictions.iloc[i]["regression"] == y_gold[i]
        error_df.loc[i, "SVM"] = predictions.iloc[i]["SVM"] == y_gold[i]
        error_df.loc[i, "MLP"] = predictions.iloc[i]["MLP"] == y_gold[i]
        error_df.loc[i, "RF"] = predictions.iloc[i]["RF"] == y_gold[i]

        pred_compare = [
            predictions.iloc[i]["regression"] == y_gold[i],
            predictions.iloc[i]["SVM"] == y_gold[i],
            predictions.iloc[i]["MLP"] == y_gold[i],
            predictions.iloc[i]["RF"] == y_gold[i],
        ]

        if pred_compare.count(True) == 4:
            error_df.loc[i, "all_true"] = True
        else:
            error_df.loc[i, "all_true"] = False

        if pred_compare.count(False) == 4:
            error_df.loc[i, "all_false"] = True
        else:
            error_df.loc[i, "all_false"] = False

        if pred_compare.count(True) == 1:
            error_df.loc[i, "one_true"] = True
        else:
            error_df.loc[i, "one_true"] = False

    return error_df


if __name__ == "__main__":
    for i in range(0, 5):
        # read in the predictions of the current run
        predictions = pd.read_csv(f"../results/run_{i}/all_predictions.csv")

        sent_ids = predictions["sent-id"]
        y_gold = predictions["gold_label"]

        error_df = get_error_df(predictions, sent_ids, y_gold)
        print(error_df.head())
        # save the error dataframe of the current run
        error_df.to_csv(f"../results/run_{i}/error_df.csv", index=False)

        pred_analysis = {
            "all_true": [int(x) for x in error_df[error_df["all_true"] == True]["sent-id"].tolist()],
            "all_false": [int(x) for x in error_df[error_df["all_false"] == True]["sent-id"].tolist()],
            "one_true": [int(x) for x in error_df[error_df["one_true"] == True]["sent-id"].tolist()],
        }

        # save the error analysis of the current run as a JSON file
        with open(f"../results/run_{i}/pred_analysis.json", 'w') as json_file:
            json.dump(pred_analysis, json_file)

