# TODO:
# HACK:
"""
=============================================================================
🚀 FUTURE OPTIMIZATION & DEPLOYMENT ROADMAP
=============================================================================
1. Fine-Tuning (LoRA):
   - Train a LoRA adapter on the base HF model (google/gemma-2-9b-it) to
     learn specific edge cases.
   - Merge the adapter with the base weights.
   - Export the merged model to a custom .gguf for deployment.

2. GPU Server Deployment (Scalability):
   - Engine: Replace llama.cpp with vLLM for maximum GPU throughput.
   - Precision: Upgrade to native bfloat16 (needs ~24GB VRAM) for max
     accuracy, or use AWQ/FP8 quantization (GPU-optimized compression).
   - Throughput: Implement Continuous Batching (feed 50+ emails at once
     via API instead of sequential loops).
=============================================================================
"""

# !/usr/bin/env python3
import sys
import os
import traceback
from dotenv import load_dotenv
from transformers import logging as hf_logging

# This automatically finds your .env file and loads the variables into the system
load_dotenv()

# Now you can safely grab the token without hardcoding it
HF_TOKEN = os.getenv("HF_TOKEN")


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
