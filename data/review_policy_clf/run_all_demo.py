# run_all_demo.py
"""
Demo script:
1) Loads sample reviews from reviews.csv ('text' column used).
2) Runs Qwen LLM zero-/few-shot prompt classification.
3) Runs zero-shot NLI classifier.
4) (Optional) Runs fine-tuned model if user has prepared 'policy_label' column and fine-tuned.
Outputs a CSV with predictions from each method.
"""

import pandas as pd
from prompt_utils import POLICY_LABELS
from llm_zero_shot_qwen import QwenPolicyClassifier
from zero_shot_nli import nli_predict_batch

def main(input_csv="reviews.csv", out_csv="predictions.csv", head_n=50):
    df = pd.read_csv(input_csv).head(head_n).copy()
    assert "text" in df.columns, "Expected a 'text' column in the CSV."

    texts = df["text"].astype(str).tolist()

    # 1) Qwen LLM (zero-/few-shot prompt classification)
    qwen = QwenPolicyClassifier()
    df["pred_qwen"] = qwen.batch_predict(texts)

    # 2) Zero-shot NLI (maps label names directly)
    df["pred_nli"] = nli_predict_batch(texts, candidate_labels=POLICY_LABELS)

    df.to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv}")

if __name__ == "__main__":
    main()
