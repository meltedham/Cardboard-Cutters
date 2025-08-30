import pandas as pd
from transformers import pipeline

def load_and_get_ai_score(file_path, output_path):
    # Load reviews
    df = pd.read_csv(file_path)
    review_col = "text" if "text" in df.columns else df.columns[0]

    # Load AI detector pipeline
    ai_detector = pipeline("text-classification", model="roberta-base-openai-detector")

    def get_ai_score(review):
        result = ai_detector(review)
        # result is a list of dicts, pick the one with label 'LABEL_1' (AI-generated)
        for r in result:
            if r['label'] == 'LABEL_1':
                return r['score'] * 100  # percentage
        # If LABEL_1 not found, return 0
        return (1 - r['score']) * 100

    # Apply to all reviews
    df['ai_generated_score'] = df[review_col].astype(str).apply(get_ai_score)

    # Save results
    df.to_csv(output_path, index=False)
    print("Saved with AI-generated scores")
    return df