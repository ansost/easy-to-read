"""Compare num_statement predictions from AMR and rule-based approaches

Prequisites:
    - results from AME based approach are available
    - results from rule based approach are available

Usage:
    python scripts/check_three_approaches.py
"""
import pandas as pd


if __name__ == "__main__":
    # change file paths here
    amr_path = "../data/metrics/test_amr.csv"
    rule_based_path = "../results/rule_predictions_test_wo_zero_statements.csv"

    # read in amr and rule-based predictions
    amr = list(pd.read_csv(amr_path)["num_statements"])
    rule_based = list(pd.read_csv(rule_based_path)["num_statements"])

    if len(amr) != len(rule_based):
        print("Lengths predictions do not match")
        exit()
    else:
        # check how amr and rule based predictions compare
        print(amr == rule_based)
        for i in range(len(amr)):
            if amr[i] != rule_based[i]:
                print(i, amr[i], rule_based[i])

