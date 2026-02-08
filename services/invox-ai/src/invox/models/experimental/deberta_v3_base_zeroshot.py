import sys
from transformers import pipeline
from invox.features.email_classification.categories import LABELS


def classify(file_path):
    # Load Model
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-zeroshot-v2.0",
        device=-1,
    )

    with open(file_path, "r") as f:
        text = f.read()[:4000]

    result = classifier(
        text, LABELS, multi_label=False, hypothesis_template="This text is about {}."
    )
    return f"tag: [{result['labels'][0]}]"


if __name__ == "__main__":
    print(classify(sys.argv[1]))