import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import spacy
from tqdm import tqdm

nlp = spacy.load("de_core_news_lg")

df = pd.read_csv("../data/train_trial_test.csv")
texts = df["phrase"].tolist()
labels = df["num_statements"].tolist()

class StatementCountDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        doc = nlp(text)
        pos_tags = [token.pos_ for token in doc]
        pos_encoding = self.encode_pos_tags(pos_tags)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'pos_tags': torch.tensor(pos_encoding, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def encode_pos_tags(self, pos_tags):
        # Simple one-hot encoding for POS tags
        pos_set = set(pos_tags)
        return [1 if pos in pos_set else 0 for pos in nlp.pipe_labels['tagger']]

class BertWithPOSFeatures(torch.nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, num_labels=num_labels)
        self.pos_encoder = torch.nn.Linear(len(nlp.pipe_labels['tagger']), 64)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size + 64, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, pos_tags):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = bert_output.last_hidden_state[:, 0, :]  # Use CLS token output
        pos_encoded = self.pos_encoder(pos_tags)
        combined_features = torch.cat((pooled_output, pos_encoded), dim=1)
        logits = self.classifier(combined_features)
        return logits

num_classes = max(labels) + 1

# Load the German BERT model and tokenizer
model_name = "bert-base-german-cased"
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
model = BertWithPOSFeatures(model_name, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model.to(device)

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

# Create datasets and dataloaders
max_length = 128
train_dataset = StatementCountDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = StatementCountDataset(val_texts, val_labels, tokenizer, max_length)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        pos_tags = batch['pos_tags'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pos_tags=pos_tags)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            pos_tags = batch['pos_tags'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pos_tags=pos_tags)
            _, preds = torch.max(outputs, dim=1)
            val_predictions.extend(preds.cpu().tolist())
            val_true_labels.extend(labels.cpu().tolist())

    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(val_true_labels, val_predictions))

torch.save(model.state_dict(), "../data/fine_tuned_german_statement_classifier_with_pos_features.pt")
tokenizer.save_pretrained("../data/fine_tuned_german_statement_tokenizer_with_pos")
