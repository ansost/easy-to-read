"""Fine-tune German BERT with POS features for statement count + span prediction.

Two-stage approach:
1. Sequence classification: BERT + POS features for statement count prediction
2. Token classification: BERT for statement span prediction (BIO-like labeling)
"""

import argparse

import torch
import pandas as pd
import numpy as np
import spacy
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel, AutoTokenizer,
    AutoModelForTokenClassification, TrainingArguments, Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


# --- Stage 1: Sequence classification with POS features ---

class StatementCountDataset(Dataset):
    """Dataset for statement count classification with POS features."""

    def __init__(self, texts, labels, tokenizer, nlp, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        doc = self.nlp(text)
        pos_tags = [token.pos_ for token in doc]
        pos_encoding = [1 if pos in set(pos_tags) else 0 for pos in self.nlp.pipe_labels["tagger"]]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "pos_tags": torch.tensor(pos_encoding, dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BertWithPOSFeatures(torch.nn.Module):
    """BERT model augmented with POS tag features for classification."""

    def __init__(self, bert_model_name, num_labels, pos_vocab_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, num_labels=num_labels)
        self.pos_encoder = torch.nn.Linear(pos_vocab_size, 64)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size + 64, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, pos_tags):
        bert_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        pooled_output = bert_output.last_hidden_state[:, 0, :]  # CLS token
        pos_encoded = self.pos_encoder(pos_tags)
        combined = torch.cat((pooled_output, pos_encoded), dim=1)
        logits = self.classifier(combined)
        return logits, pooled_output


def train_sequence_classifier(data_path, model_name="bert-base-german-cased",
                               num_epochs=8, batch_size=16, lr=2e-5, max_length=128):
    """Train BERT + POS sequence classifier for statement count."""
    nlp = spacy.load("de_core_news_sm")

    df_train = pd.read_csv(f"{data_path}/train.csv")
    df_trial = pd.read_csv(f"{data_path}/trial.csv")
    df_augmented = pd.read_csv(f"{data_path}/augmented.csv")
    df_all = pd.concat([df_train, df_trial, df_augmented])
    df_all = df_all.drop_duplicates(subset="sent-id", keep="first")

    df = df_all[df_all["num_statements"] > 0][["phrase", "num_statements"]]
    texts = df["phrase"].tolist()
    labels = df["num_statements"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.05, random_state=42
    )

    num_classes = int(max(train_labels) + 1)
    pos_vocab_size = len(nlp.pipe_labels["tagger"])
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertWithPOSFeatures(model_name, num_classes, pos_vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_dataset = StatementCountDataset(train_texts, train_labels, tokenizer, nlp, max_length)
    val_dataset = StatementCountDataset(val_texts, val_labels, tokenizer, nlp, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            pos_tags = batch["pos_tags"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs, _ = model(input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids, pos_tags=pos_tags)
            loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Average training loss: {total_loss / len(train_dataloader):.4f}")

        model.eval()
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                outputs, _ = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                    pos_tags=batch["pos_tags"].to(device),
                )
                _, preds = torch.max(outputs, dim=1)
                val_predictions.extend(preds.cpu().tolist())
                val_true_labels.extend(batch["labels"].tolist())

        print(f"Validation Accuracy: {accuracy_score(val_true_labels, val_predictions):.4f}")

    print("\nClassification Report:")
    print(classification_report(val_true_labels, val_predictions))
    return model, tokenizer, nlp


# --- Stage 2: Token classification for statement spans ---

def read_sentence(tokenized):
    """Parse tokenized sentence format '0:=word1 1:=word2' into (max_idx, tokens)."""
    tokens = tokenized.split(" ")
    strip_tokens = [t.split(":=")[1] for t in tokens]
    max_idx = int(tokens[-1].split(":=")[0])
    return max_idx, strip_tokens


def construct_spans(max_idx, statement_span):
    """Convert statement spans to per-token labels.

    Input: max_idx=10, statement_span=[[2,4],[6,7,8]]
    Output: [0, 0, 1, 0, 1, 0, 2, 2, 2, 0, 0]
    """
    spans = [0] * (max_idx + 1)
    if not isinstance(statement_span, (list, tuple)):
        return spans
    for c, span in enumerate(statement_span, start=1):
        for i in span:
            spans[i] = c
    return spans


def predicted_statement_spans(pred):
    """Convert per-token predictions back to statement spans."""
    if 1 not in pred:
        return None
    pred_spans = []
    for c in range(1, 20):
        if c in pred:
            pred_spans.append([i for i, label in enumerate(pred) if label == c])
    return pred_spans


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize and align span labels with BERT subword tokens."""
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_token_classifier(data_path, checkpoint="bert-base-german-cased", num_epochs=10):
    """Train token classifier for statement span prediction."""
    import evaluate
    from datasets import Dataset as HFDataset, DatasetDict

    df_train = pd.read_csv(f"{data_path}/train.csv")
    df_trial = pd.read_csv(f"{data_path}/trial.csv")
    df_augmented = pd.read_csv(f"{data_path}/augmented.csv")
    df_test = pd.read_csv(f"{data_path}/test.csv")
    df_eval = pd.read_csv(f"{data_path}/eval.csv")

    df_all = pd.concat([df_train, df_trial, df_augmented, df_test, df_eval])
    df_all = df_all.drop_duplicates(subset="sent-id", keep="first")
    df_all = df_all.replace(np.nan, None)
    df_all["statement_spans"] = df_all["statement_spans"].apply(
        lambda x: None if x is None else eval(x)
    )
    df_all["tokenized"] = df_all["phrase_tokenized"].apply(read_sentence)
    df_all["span_labels"] = df_all.apply(
        lambda row: construct_spans(row["tokenized"][0], row["statement_spans"]), axis=1
    )

    # Assign split types
    df_all["type"] = "augmented"
    df_all.loc[df_all["sent-id"].isin(df_trial["sent-id"]), "type"] = "trial"
    df_all.loc[df_all["sent-id"].isin(df_train["sent-id"]), "type"] = "train"
    df_all.loc[df_all["sent-id"].isin(df_test["sent-id"]), "type"] = "test"
    df_all.loc[df_all["sent-id"].isin(df_eval["sent-id"]), "type"] = "eval"

    df_mine = df_all[["tokenized", "span_labels", "type", "sent-id", "num_statements"]]
    df_mine["tokenized"] = df_mine["tokenized"].apply(lambda x: x[1])
    df_mine = df_mine.rename(columns={"tokenized": "tokens", "span_labels": "tags"})

    df_mine_train = df_mine[df_mine["type"].isin(["train", "trial"])]
    df_mine_train = df_mine_train[df_mine_train["num_statements"] != 0]
    df_mine_train, df_mine_val = train_test_split(df_mine_train, test_size=0.05, random_state=42)

    dataset_dict = DatasetDict({
        "train": HFDataset.from_pandas(df_mine_train),
        "validation": HFDataset.from_pandas(df_mine_val),
        "test": HFDataset.from_pandas(df_mine[df_mine["type"] == "test"]),
        "finaleval": HFDataset.from_pandas(df_mine[df_mine["type"] == "eval"]),
    })

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_dataset = dataset_dict.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True
    )

    label_list = [f"{i}span" for i in range(10)]
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir="Germeval24StageTask2",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt"),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    return trainer, tokenized_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data", help="Path to data directory")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--epochs-stage1", type=int, default=8)
    parser.add_argument("--epochs-stage2", type=int, default=10)
    args = parser.parse_args()

    if args.stage in ("1", "both"):
        print("Stage 1: Sequence Classification")
        model, tokenizer, nlp = train_sequence_classifier(
            args.data_path, num_epochs=args.epochs_stage1
        )

    if args.stage in ("2", "both"):
        print("Stage 2: Token Classification")
        trainer, dataset = train_token_classifier(
            args.data_path, num_epochs=args.epochs_stage2
        )
