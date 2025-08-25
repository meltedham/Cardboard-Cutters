# src/utils.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Calculate evaluation metrics (Accuracy, Precision, Recall, F1)
def evaluate_model(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return accuracy, precision, recall, f1
