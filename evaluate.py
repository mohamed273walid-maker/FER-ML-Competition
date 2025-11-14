import pandas as pd
from sklearn.metrics import accuracy_score
import os

# Ensure submissions folder exists
os.makedirs("submissions", exist_ok=True)

def evaluate(submission_path="submission/submission.csv",
             labels_path="evaluation/test_labels.csv"):
    """
    Evaluates a participant's submission against hidden labels.
    """
    # Load submission
    submission = pd.read_csv(submission_path)

    # Load hidden labels
    hidden_labels = pd.read_csv(labels_path)

    # Merge on ID to make sure we align
    merged = pd.merge(hidden_labels, submission, on="id")

    # Compute accuracy
    acc = accuracy_score(merged['emotion'], merged['predicted_label'])
    print(f"Participant Submission Accuracy: {acc*100:.2f}%")
    return acc

if __name__ == "__main__":
    evaluate()