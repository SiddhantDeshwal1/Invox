import sys
from transformers import pipeline
from invox.features.email_classification.categories import LABELS


def classify(file_path):
    classifier = pipeline(
        "zero-shot-classification", model="roberta-large-mnli", device=-1
    )

    with open(file_path, "r") as f:
        text = f.read()[:2000]

    result = classifier(text, LABELS, multi_label=False)
    return f"tag: [{result['labels'][0]}]"


if __name__ == "__main__":
    print(classify(sys.argv[1]))