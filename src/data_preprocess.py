import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from wordfreq import zipf_frequency

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+|\.com", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

def remove_non_english(text, threshold=3.0):
    return " ".join([w for w in text.split() if zipf_frequency(w, "en") >= threshold])

def preview_cleaning(df, review_col, output_path, n=20):
    """Show sample of original vs cleaned reviews."""
    preview_rows = []
    print("\n=== Preview of Cleaning ===")
    for i, row in df.head(n).iterrows():
        original = row[review_col]
        cleaned = remove_non_english(remove_stopwords(clean_text(original)))
        preview_rows.append({"original": original, "cleaned": cleaned})
        print(f"\nOriginal: {original}\nCleaned : {cleaned}")

    preview_df = pd.DataFrame(preview_rows)
    preview_file = output_path.replace(".csv", "_preview.csv")
    preview_df.to_csv(preview_file, index=False)
    print(f"\nSaved cleaning preview to {preview_file}")

def preprocess_reviews(file_path, output_path):
    df = pd.read_csv(file_path)
    review_col = "text" if "text" in df.columns else df.columns[0]

    df = df.dropna(subset=[review_col])
    df = df.drop_duplicates(subset=[review_col])

    preview_cleaning(df, review_col, output_path, n=20)

    df[review_col] = df[review_col].apply(clean_text)
    df[review_col] = df[review_col].apply(remove_stopwords)
    df[review_col] = df[review_col].apply(remove_non_english)

    df = df[df[review_col].str.split().str.len() >= 3]

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path} with {len(df)} reviews.")
    return df
