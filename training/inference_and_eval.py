import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Label mappings (must match training)
label2id = {
    "valid review": 0,
    "advertisement": 1,
    "irrelevant": 2,
    "rant without visit": 3,
}
id2label = {v: k for k, v in label2id.items()}

# Load validation data
df = pd.read_csv("../data/reviews_labeled.csv")
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].map(label2id)
_, val_df = train_test_split(df, test_size=0.2, random_state=42)
val_dataset = Dataset.from_pandas(val_df)

# Load model and tokenizer
model_dir = "./results/final_model"  # Change if you saved elsewhere
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

# Tokenize validation set
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

val_dataset = val_dataset.map(tokenize, batched=True)
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Evaluate
trainer = Trainer(model=model)
results = trainer.evaluate(val_dataset)
print("Validation results:", results)

# Predict on new text
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1).item()
    return id2label[pred]

# Example usage
sample_text = "The food was great and the staff were friendly."
print("Prediction for sample text:", predict(sample_text))