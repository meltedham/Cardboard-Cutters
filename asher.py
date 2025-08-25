import re
import pandas as pd
from transformers import pipeline

df = pd.read_csv("data/reviews.csv")

print("Columns in dataset:", df.columns)
review_col = "text" if "text" in df.columns else df.columns[0]

reviews = df[review_col].dropna().sample(10, random_state=42).tolist()

labels = ["advertisement", "irrelevant", "rant_without_visit", "valid_review"]

print("Loading Hugging Face model... (first time may take a minute)")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def contains_ad(review: str) -> bool:
    return bool(re.search(r"(http|www|\.com|discount|promo)", review.lower()))

for review in reviews:
    result = classifier(review, labels)
    predicted_label = result["labels"][0]
    confidence = result["scores"][0]

    if contains_ad(review):
        predicted_label = "advertisement (regex rule)"

    print("\n---")
    print(f"Review: {review}")
    print(f"Predicted: {predicted_label} (confidence {confidence:.2f})")
