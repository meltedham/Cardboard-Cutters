import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm import tqdm
import torch

# Label mappings (must match training)
label2id = {
    "valid review": 0,
    "advertisement": 1,
    "irrelevant": 2,
    "rant without visit": 3,
}
id2label = {v: k for k, v in label2id.items()}

# Load model and tokenizer
model_dir = "./results/final_model"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Load reviews
input_csv = "../data/reviews_cleaned_with_ai_score3.csv"
df = pd.read_csv(input_csv)

# Predict labels
def predict_label(text):
    inputs = tokenizer(str(text), return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()
    return pred

tqdm.pandas()
df["predicted_label"] = df["text"].progress_apply(predict_label)

# Filter for valid reviews (label 0)
valid_reviews = df[df["predicted_label"] == 0]

# Save only valid reviews to new CSV
output_csv = "../data/valid_reviews.csv"
valid_reviews.to_csv(output_csv, index=False)
print(f"Saved {len(valid_reviews)} valid reviews to {output_csv}")