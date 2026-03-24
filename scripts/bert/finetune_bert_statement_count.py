"""Fine-tune German BERT for statement count prediction.

Trains a BertForSequenceClassification model to predict the number of statements.
"""

import argparse

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class StatementCountDataset(Dataset):
    """Dataset for statement count classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_model(data_path, model_name, save_path, num_epochs=50, batch_size=16,
                lr=2e-5, max_length=128, test_size=0.1):
    """Train the BERT model for statement count prediction."""
    df = pd.read_csv(data_path)
    texts = df["phrase"].tolist()
    labels = df["num_statements"].tolist()
    num_classes = max(labels) + 1

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )

    train_dataset = StatementCountDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = StatementCountDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
            outputs.loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                val_predictions.extend(preds.cpu().tolist())
                val_true_labels.extend(batch_labels.cpu().tolist())

        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(val_true_labels, val_predictions))

    model.save_pretrained(save_path + "_model")
    tokenizer.save_pretrained(save_path + "_tokenizer")
    return model, tokenizer


def predict_statement_count(text, model, tokenizer, device, max_length=128):
    """Predict the number of statements in a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction


def evaluate_on_test(model, tokenizer, eval_path, labels_path, device):
    """Evaluate model on test set and return accuracy."""
    df = pd.read_csv(eval_path)
    test_texts = df["phrase"].tolist()
    y_df = pd.read_csv(labels_path)
    test_labels = y_df["num_statements"].tolist()

    predicted = [predict_statement_count(text, model, tokenizer, device) for text in test_texts]
    acc = accuracy_score(test_labels, predicted)
    print(f"Test accuracy: {acc:.4f}")
    return predicted, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for statement count prediction")
    parser.add_argument("--data", default="data/train_trial_test.csv", help="Training data CSV")
    parser.add_argument("--model-name", default="bert-base-german-cased", help="Pre-trained model name")
    parser.add_argument("--save-path", default="models/fine_tuned_german_statement", help="Save path prefix")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model, tokenizer = train_model(args.data, args.model_name, args.save_path,
                                   num_epochs=args.epochs, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_text = "Das ist ein Beispieltext. Er enthält drei Sätze. Hier ist der letzte."
    count = predict_statement_count(test_text, model, tokenizer, device)
    print(f"\nTest: '{test_text}' -> {count} statements")
