#!/usr/bin/env python3
import sys
import os
import traceback

# ===== GLOBAL TOKEN & VERBOSE CONFIG =====
# Removed all silencing logic. Hugging Face will now log natively.
os.environ["HF_TOKEN"] = "REMOVED"

# Force transformers to show info (loading weights, etc.)
from transformers import logging as hf_logging

hf_logging.set_verbosity_info()

print("\n" + "=" * 50)
print("⚙️  INITIALIZING INVOX AI CLASSIFICATION PIPELINE")
print("=" * 50)

try:
    print("[INIT] Loading Spam Detection Module...")
    from invox.features.email_classification.spam_detection import check_is_spam

    print("[INIT] Loading Category Detection Module...")
    from invox.features.email_classification.category_detection import (
        classify_hierarchical,
    )

    print("[INIT] All modules loaded successfully!\n")
except Exception as e:
    print("cat: Error\nsubcat: ImportFailure")
    print(f"\n--- FATAL IMPORT ERROR ---\n{traceback.format_exc()}", file=sys.stderr)
    sys.exit(1)


def process_email_classification(email_text: str) -> str:
    try:
        print("\n" + "-" * 40)
        print("🛡️  RUNNING SPAM DETECTION")
        print("-" * 40)

        is_spam = check_is_spam(email_text)

        print(
            f"✅ Successfully executed Spam Detection. Result: {'SPAM' if is_spam else 'HAM'}"
        )

        if is_spam:
            print("🛑 Halting pipeline early due to Spam.")
            return "cat: Spam\nsubcat: Spam"

        print("\n" + "-" * 40)
        print("🧠 RUNNING CATEGORY DETECTION (GEMMA 2)")
        print("-" * 40)

        parent, child = classify_hierarchical(email_text)

        print("✅ Successfully executed Category Detection.")

        print("\n" + "=" * 50)
        print("🎯 FINAL OUTPUT FORMATTED")
        print("=" * 50)
        return f"cat: {parent}\nsubcat: {child}"

    except Exception as e:
        print(
            f"\n--- FATAL PIPELINE ERROR ---\n{traceback.format_exc()}", file=sys.stderr
        )
        return "cat: Error\nsubcat: RuntimeFailure"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("cat: Error\nsubcat: NoInputFile")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print("cat: Error\nsubcat: FileNotFound")
        sys.exit(1)

    try:
        print(f"\n📂 Reading input file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        final_output = process_email_classification(content)

        # This clean output string goes right to the terminal at the very end
        print("\n" + final_output + "\n")

    except Exception as e:
        print(f"\n--- FATAL FILE ERROR ---\n{traceback.format_exc()}", file=sys.stderr)
        print("cat: Error\nsubcat: RuntimeFailure")
