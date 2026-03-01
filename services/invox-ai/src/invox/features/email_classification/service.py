#!/usr/bin/env python3
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

import sys
import os
import time
import traceback
from dotenv import load_dotenv
from transformers import logging as hf_logging

# This automatically finds your .env file and loads the variables into the system
load_dotenv()

# Now you can safely grab the token without hardcoding it
HF_TOKEN = os.getenv("HF_TOKEN")

hf_logging.set_verbosity_info()

print("\n" + "=" * 50, file=sys.stderr)
print("⚙️  INITIALIZING INVOX AI CLASSIFICATION PIPELINE", file=sys.stderr)
print("=" * 50, file=sys.stderr)

try:
    print("[INIT] Loading Spam Detection Module...", file=sys.stderr)
    from invox.features.email_classification.spam_detection import check_is_spam

    print("[INIT] Loading Category Detection Module...", file=sys.stderr)
    from invox.features.email_classification.category_detection import (
        classify_hierarchical,
    )

    print("[INIT] All modules loaded successfully!\n", file=sys.stderr)
except Exception as e:
    print(f"\n--- FATAL IMPORT ERROR ---\n{traceback.format_exc()}", file=sys.stderr)
    sys.exit(1)


def process_email_classification(email_text: str) -> tuple[str, str]:
    try:
        is_spam = check_is_spam(email_text)

        if is_spam:
            return "Spam", "Spam"

        parent, child = classify_hierarchical(email_text)
        return parent, child

    except Exception as e:
        print(
            f"\n--- FATAL PIPELINE ERROR ---\n{traceback.format_exc()}", file=sys.stderr
        )
        return "Error", "RuntimeFailure"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("RESULT|Error|NoInputFile|0.00")
        sys.exit(1)

    # Grab ALL files passed from the Bash script
    file_paths = sys.argv[1:]

    # Loop through the files without ever unloading the GGUF model from RAM
    for file_path in file_paths:
        filename = os.path.basename(file_path)

        if not os.path.exists(file_path):
            print(f"RESULT|{filename}|Error|FileNotFound|0.00", flush=True)
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            start_time = time.time()
            final_parent, final_child = process_email_classification(content)
            elapsed_time = time.time() - start_time

            # Print the exact string format the new bash script expects
            print(
                f"RESULT|{filename}|{final_parent}|{final_child}|{elapsed_time:.3f}",
                flush=True,
            )

        except Exception as e:
            print(
                f"\n--- FATAL FILE ERROR ---\n{traceback.format_exc()}", file=sys.stderr
            )
            print(f"RESULT|{filename}|Error|RuntimeFailure|0.00", flush=True)
