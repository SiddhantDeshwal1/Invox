import sys
from transformers import pipeline

print("  -> [Spam Gate] Initializing BERT-tiny model weights...", file=sys.stderr)

spam_classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-enron-spam-detection",
    device_map="auto",
    truncation=True,
    max_length=512,
)

print("  -> [Spam Gate] Model loaded into VRAM successfully.", file=sys.stderr)


def check_is_spam(email_text: str) -> bool:
    """
    Evaluates email text to determine if it is spam.
    Returns True if spam, False if ham.
    """
    print("  -> [Spam Gate] Analyzing text patterns...", file=sys.stderr)
    result = spam_classifier(email_text)[0]

    label = result["label"].lower()
    score = result["score"]

    print(
        f"  -> [Spam Gate] Inference completed. Raw Label: {label}, Confidence: {score:.4f}",
        file=sys.stderr,
    )

    return label == "spam" or label == "label_1"
