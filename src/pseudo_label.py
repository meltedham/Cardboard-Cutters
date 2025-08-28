import pandas as pd
from transformers import pipeline

INPUT = "../data/google_reviews_cleaned.csv"   # use your cleaned file
OUTPUT = "../data/reviews_labeled.csv"
TEXT_COL = "text"  # change if your column is named differently

LABELS = ["advertisement", "irrelevant", "rant_without_visit", "valid_review"]

def main():
    df = pd.read_csv(INPUT)
    assert TEXT_COL in df.columns, f"Missing column {TEXT_COL}"
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # TIP: start with a subset for speed (e.g., 2000 rows), then scale up
    df = df.sample(min(len(df), 2000), random_state=42).reset_index(drop=True)

    preds = []
    for t in df[TEXT_COL].astype(str):
        r = clf(t, LABELS)
        preds.append(r["labels"][0])  # top label

    df["label"] = preds
    df.to_csv(OUTPUT, index=False)
    print(f"Saved pseudo-labeled data to {OUTPUT} (rows: {len(df)})")

if __name__ == "__main__":
    main()
