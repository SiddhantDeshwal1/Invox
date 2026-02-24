import sys
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from invox.features.email_classification.categories import (
    get_parent_labels,
    get_child_labels,
)

print(
    "  -> [Gemma Router] Locating optimized GGUF weights for CPU inference...",
    file=sys.stderr,
)

# Download the pre-quantized 4-bit GGUF model specifically built for CPUs/iGPUs.
# This is a 5.6GB file that will fit perfectly inside your 16GB of RAM.
model_path = hf_hub_download(
    repo_id="bartowski/gemma-2-9b-it-GGUF",
    filename="gemma-2-9b-it-Q4_K_M.gguf",
    token=os.environ.get("HF_TOKEN", "REMOVED"),
)

print(
    "  -> [Gemma Router] Booting up Llama.cpp engine on Ryzen CPU...", file=sys.stderr
)

# Initialize the blazing-fast CPU engine
llm = Llama(
    model_path=model_path,
    n_ctx=2048,  # Context window size
    n_threads=4,  # Optimized for your Ryzen 5 (4 cores)
    verbose=False,  # Suppress massive C++ logs
)


def ask_gemma(prompt: str, step_label: str) -> str:
    print(
        f"    -> [Gemma Inference] Generating {step_label} classification...",
        file=sys.stderr,
    )

    # Gemma 2 Chat Template format
    formatted_prompt = (
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    )

    output = llm(
        formatted_prompt,
        max_tokens=15,
        stop=["<end_of_turn>", "\n"],
        temperature=0.0,  # Deterministic logic
    )

    raw_out = output["choices"][0]["text"]
    clean_out = raw_out.strip().replace(".", "").replace('"', "").replace("'", "")

    print(
        f"    -> [Gemma Inference] Raw output captured: '{clean_out}'", file=sys.stderr
    )
    return clean_out


def classify_hierarchical(email_text: str) -> tuple[str, str]:
    print("  -> [Gemma Router] Formatting payload...", file=sys.stderr)
    text_snippet = email_text[
        :1500
    ]  # Slightly shorter to ensure it fits in 2048 context

    parents = get_parent_labels()
    if "Spam" in parents:
        parents.remove("Spam")

    # --- Parent Step ---
    print(
        f"  -> [Gemma Router] Step 1: Asking Gemma for Parent Category...",
        file=sys.stderr,
    )
    p_prompt = f"""You are an expert email routing AI. 
Analyze the email below and classify it into exactly ONE of these categories: {parents}.
Output ONLY the exact category name from the list. Do not explain.

Email Content:
{text_snippet}

Category Name:"""

    raw_parent = ask_gemma(p_prompt, "Parent")

    top_parent = next(
        (p for p in parents if p.lower() in raw_parent.lower()), parents[0]
    )
    print(
        f"  -> [Gemma Router] Step 1 resolved. Mapped to: [{top_parent}]",
        file=sys.stderr,
    )

    # --- Child Step ---
    children = get_child_labels(top_parent)
    if not children:
        print(
            "  -> [Gemma Router] No sub-categories configured for this parent. Bypassing Step 2.",
            file=sys.stderr,
        )
        return top_parent, "General"

    print(
        f"  -> [Gemma Router] Step 2: Asking Gemma for Child Category within '{top_parent}'...",
        file=sys.stderr,
    )
    c_prompt = f"""You are an expert email routing AI. 
This email belongs to the '{top_parent}' category. 
Pick the most specific sub-category from this list: {children}.
Output ONLY the exact sub-category name from the list. Do not explain.

Email Content:
{text_snippet}

Sub-category Name:"""

    raw_child = ask_gemma(c_prompt, "Child")

    top_child = next(
        (c for c in children if c.lower() in raw_child.lower()), children[0]
    )
    print(
        f"  -> [Gemma Router] Step 2 resolved. Mapped to: [{top_child}]",
        file=sys.stderr,
    )

    return top_parent, top_child
