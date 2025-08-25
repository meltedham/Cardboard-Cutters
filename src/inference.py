# src/inference.py
from transformers import pipeline

# Load a pre-trained model (e.g., fine-tuned BERT model)
def load_trained_model(model_path):
    classifier = pipeline("zero-shot-classification", model=model_path)
    return classifier

# Perform inference (classification) on a single review
def classify_review(classifier, review, labels):
    result = classifier(review, candidate_labels=labels)
    return result
