import sys
import numpy as np
from sentence_transformers import CrossEncoder


def classify(file_path):
    model = CrossEncoder("cross-encoder/nli-distilroberta-base")

    labels = [
        "Interview Invitation",
        "Job Application Rejection",
        "Job Offer & Contract",
    ]

    with open(file_path, "r") as f:
        text = f.read()[:2000]

    # Formulate pairs for comparison
    pairs = [[text, f"This is an email about {label}"] for label in labels]

    # Scores: [Contradiction, Entailment, Neutral]
    scores = model.predict(pairs)

    # Index 1 is 'Entailment' (The match score)
    entailment_scores = scores[:, 1]
    best_idx = entailment_scores.argmax()

    return f"tag: [{labels[best_idx]}]"


if __name__ == "__main__":
    print(classify(sys.argv[1]))
