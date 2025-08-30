import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    if pd.isnull(text):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|\.com\S*", "", text)
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable characters
    text = "".join(c for c in text if c.isprintable())
    return text.strip()

def preview_cleaning(df, review_col, output_path, n=20):
    preview_rows = []
    print("\n=== Preview of Cleaning ===")
    for i, row in df.head(n).iterrows():
        original = row[review_col]
        cleaned = clean_text(original)
        preview_rows.append({"original": original, "cleaned": cleaned})
        print(f"\nOriginal: {original}\nCleaned : {cleaned}")

    # Convert Path to string for string replacement
    preview_file = Path(str(output_path).replace(".csv", "_preview.csv"))
    preview_df = pd.DataFrame(preview_rows)
    preview_df.to_csv(preview_file, index=False)
    print(f"\nSaved cleaning preview to {preview_file}")

def preprocess_reviews(file_path, output_path):
    df = pd.read_csv(file_path)
    review_col = "text" if "text" in df.columns else df.columns[0]

    # Remove rows with missing values in the review column
    df = df.dropna(subset=[review_col])
    df = df.drop_duplicates(subset=[review_col])

    preview_cleaning(df, review_col, output_path, n=20)

    df[review_col] = df[review_col].apply(clean_text)

    # Keep reviews with at least 3 words
    df = df[df[review_col].str.split().str.len() >= 3]

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path} with {len(df)} reviews.")
    return df

if __name__ == "__main__":
    input_file = "../data/reviews.csv"      # Change this to your input CSV file
    output_file = "../data/reviews_cleaned.csv"    # Change this to your desired output CSV file
    preprocess_reviews(input_file, output_file)