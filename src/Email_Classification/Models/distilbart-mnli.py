import sys
from transformers import pipeline


def classify(file_path):
    classifier = pipeline(
        "zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=-1
    )

    # Use the same 50 labels list here...
    labels = [
        "Job Offer & Contract",
        "Bank Account Statement",
        "Online Order Confirmation",
    ]  # Shortened for brevity

    with open(file_path, "r") as f:
        text = f.read()[:2000]

    result = classifier(text, labels, multi_label=False)
    return f"tag: [{result['labels'][0]}]"


if __name__ == "__main__":
    print(classify(sys.argv[1]))
