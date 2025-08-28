import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

DATA = "../data/reviews_labeled.csv"     # from step 1 (or your human-labeled file)
TEXT_COL = "text"
LABEL_COL = "label"
MODEL_OUT = "../models/baseline_lr.joblib"

def main():
    df = pd.read_csv(DATA)
    assert TEXT_COL in df.columns and LABEL_COL in df.columns

    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(str)

    # 60/40 split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(pipe, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
