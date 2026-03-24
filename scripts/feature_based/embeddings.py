"""Get embeddings from German BERT model.
Adds a new column 'embeddings' to the dataset, containing the embeddings of the 'phrase' column.

Usage:  
    python embeddings.py --dataset <dataset>
"""

import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM


def encode(column: str, tokenizer):
    encoding = tokenizer.batch_encode_plus(
        column,  # List of input texts
        padding=True,  # Pad to the maximum sequence length
        truncation=True,  # Truncate to the maximum sequence length if necessary
        return_tensors="pt",  # Return PyTorch tensors
        add_special_tokens=True,  # Add special tokens CLS and SEP
    )
    return encoding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Any .csv file in data/metrics/, that includes a column 'num_statements'.",
    )
    args = parser.parse_args()

    data = pd.read_csv(f"../data/metrics/{args.dataset}_basics.csv")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-german-cased")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-german-cased")

    encodings = encode(data["phrase"].to_list(), tokenizer)
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state

    data["embeddings"] = word_embeddings
    data.to_csv(f"../data/metrics/{args.dataset}.csv", index=False)


if __name__ == "__main__":
    main()
