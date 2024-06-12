"""Generate predicted statement spans using rules.

Usage: 
    rules.py --dataset <dataset>
"""

import argparse
import pandas as pd
from  tqdm.auto import tqdm

from annotate import count_statements
from get_statement_spans import get_statement_spans

tqdm.pandas()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to the dataset")
    args = parser.parse_args()

    df = pd.read_csv(f"../data/metrics/{args.dataset}_basics.csv")
    df["predicted_num_statements"] = df["phrase"].progress_apply(count_statements)
    df["predicted_spans"] = df["phrase"].progress_apply(get_statement_spans)

    df["correct_num_statements"] = df.apply(lambda x: x["predicted_num_statements"] == len(x["predicted_spans"]), axis=1)
    df["correct_spans"] = df.apply(lambda x: x["predicted_spans"] == x["statement_spans"], axis=1)
    print(f"Num statements ccuracy: {df['correct_num_statements'].sum() / len(df)}")
    print(f"Span accuracy: {df['correct_spans'].sum() / len(df)}")

    # only save columns that are needed for evaluation
    df = df[["sent-id", "phrase", "num_statements", "predicted_num_statements", "statement_spans", "predicted_spans"]]
    df.to_csv(f"../results/rule_predictions_{args.dataset}.csv", index=False)

