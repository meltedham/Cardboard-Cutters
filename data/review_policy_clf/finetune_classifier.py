# finetune_classifier.py
"""
Fine-tune a text classifier on your policy labels (supervised).
Assumes your CSV has columns: 'text' and 'policy_label'.
We intentionally do not use 'evaluation_strategy' in TrainingArguments.
We run a manual evaluation after training instead.
"""

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from typing import List, Dict 

@dataclass
class TextLabelDataset(Dataset):
    encodings: Dict[str, List[int]]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def load_and_split(csv_path: str, label_col: str = "policy_label", test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(csv_path)
    assert "text" in df.columns, f"'text' column missing in {csv_path}"
    assert label_col in df.columns, f"'{label_col}' column missing in {csv_path}"
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label_col])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def encode(tokenizer, texts, max_length=192):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length)

def finetune(csv_path: str, label_col: str = "policy_label", model_name: str = "distilbert-base-uncased",
             output_dir: str = "./policy_model", epochs: int = 3, batch_size: int = 16) -> str:
    train_df, val_df = load_and_split(csv_path, label_col=label_col)
    le = LabelEncoder().fit(train_df[label_col])
    num_labels = len(le.classes_)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = encode(tokenizer, train_df["text"].tolist())
    val_enc = encode(tokenizer, val_df["text"].tolist())

    train_ds = TextLabelDataset(train_enc, le.transform(train_df[label_col]).tolist())
    val_ds = TextLabelDataset(val_enc, le.transform(val_df[label_col]).tolist())

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        load_best_model_at_end=False,
        logging_steps=20,
        save_total_limit=2,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=None)
    trainer.train()
    model.eval()
    with torch.inference_mode():
        logits = trainer.model(**{k: torch.tensor(v) for k, v in val_enc.items()}).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_true = le.transform(val_df[label_col]).astype(int)
    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, target_names=list(le.classes_), digits=4)
    print("Validation accuracy:", acc)
    print(report)

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    pd.Series(le.classes_).to_csv(f"{output_dir}/label_mapping.csv", index=False, header=False)
    return output_dir
