import joblib
import pandas as pd

MODEL = "models/baseline_lr.joblib"
INPUT = "data/google_reviews_cleaned.csv"   # or any new file of raw reviews
TEXT_COL = "text"

def predict_batch(csv_path, n=10):
    pipe = joblib.load(MODEL)
    df = pd.read_csv(csv_path)
    col = TEXT_COL if TEXT_COL in df.columns else df.columns[0]
    texts = df[col].dropna().sample(min(n, len(df)), random_state=42).astype(str)
    preds = pipe.predict(texts)
    for t, p in zip(texts, preds):
        print("\n---")
        print("Review:", t)
        print("Pred  :", p)

if __name__ == "__main__":
    predict_batch(INPUT, n=10)
