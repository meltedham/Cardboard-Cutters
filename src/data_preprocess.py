import pandas as pd
import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+|\.com", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

def preprocess_reviews(file_path, output_path):
    df = pd.read_csv(file_path)

    review_col = "text" if "text" in df.columns else df.columns[0]

    df = df.dropna(subset=[review_col])
    df = df.drop_duplicates(subset=[review_col])

    df[review_col] = df[review_col].apply(clean_text)
    df[review_col] = df[review_col].apply(remove_stopwords)

    df = df[df[review_col].str.split().str.len() >= 3]

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path} with {len(df)} reviews.")
    return df

if __name__ == "__main__":
    preprocess_reviews("data/reviews.csv", "data/google_reviews_cleaned.csv")
