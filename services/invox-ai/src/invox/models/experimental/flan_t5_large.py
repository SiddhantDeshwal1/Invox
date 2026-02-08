import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from invox.features.email_classification.categories import LABELS


def classify(file_path):
    model_name = "google/flan-t5-large"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use GPU if available, otherwise CPU
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map)

    try:
        with open(file_path, "r") as f:
            # T5 has a token limit of 512, but can handle slightly more input safely.
            # We truncate char count to 2500 to stay within safe bounds.
            text = f.read()[:2500]
    except FileNotFoundError:
        return "Error: File not found."

    # Join labels for the prompt
    labels_str = ", ".join(LABELS)

    # --- THE FIX: Use Triple Quotes (""") for multi-line strings ---
    prompt = f"""Classify the email below into one of these categories: 
{labels_str}

Email:
{text}

Category:"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return f"tag: [{prediction}]"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flan-t5-large.py <email.txt>")
    else:
        print(classify(sys.argv[1]))

