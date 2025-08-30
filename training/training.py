import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset

def train_review_classifier(
    data_path="../data/reviews_labeled.csv",
    model_save_path="./results/final_model",
    num_train_epochs=3,
    batch_size=8,
    random_state=42
):
    # Load data
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"])

    # Encode labels
    label2id = {
        "valid review": 0,
        "advertisement": 1,
        "irrelevant": 2,
        "rant without visit": 3,
    }
    id2label = {v: k for k, v in label2id.items()}
    df["label"] = df["label"].map(label2id)

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=random_state)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenization function
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="training/results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir="./logs",
        logging_steps=50,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train!
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

# Optional: Only run if called directly
if __name__ == "__main__":
    train_review_classifier()