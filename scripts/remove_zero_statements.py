"""
Script to remove zero-statements.
FIXME: add me to the pipeline script.

Usage:
    remove_zero_statements.py --split=<s>

Options:
    --split=<s>                Input file (train, test, dev)
"""
from docopt import docopt
import pandas as pd

def remove_zero_statements(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename)
    data = data[data["num_statements"] !=0]

    return data

if __name__ == "__main__":
    args = docopt(__doc__)
    split = args["--split"]
    input_file = "../data/" + split + ".csv"
    df = remove_zero_statements(input_file)
    df.to_csv("../data/" + split + "_wo_zero_statements.csv")
