# zero_shot_nli.py
"""
Zero-shot classification using an NLI model (fast, no prompt parsing).
Maps candidate labels to hypotheses and lets the model score entailment.
"""

from typing import List
from transformers import pipeline

def build_zero_shot_classifier(model_name: str = "facebook/bart-large-mnli"):
    clf = pipeline("zero-shot-classification", model=model_name)
    return clf

def nli_predict_batch(texts: List[str], candidate_labels: List[str]):
    clf = build_zero_shot_classifier()
    results = clf(texts, candidate_labels=candidate_labels, multi_label=False)
    if isinstance(results, dict):
        return [results["labels"][0]]
    return [r["labels"][0] for r in results]
