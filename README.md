# Invox

AI-powered email classification service that automatically categorizes incoming emails using a two-stage hierarchical pipeline.

## Overview

Invox processes raw email text through a **spam gate** followed by **hierarchical category detection**, assigning each email a parent category and subcategory from a rich taxonomy of 10 parent categories and ~100 subcategories.

```
Email Input → Spam Detection (BERT-tiny) → Category Detection (Gemma 2 9B) → (parent, subcat)
```

## Architecture

### Pipeline

| Stage | Module | Model | Purpose |
|-------|--------|-------|---------|
| 1 | `spam_detection.py` | `mrm8488/bert-tiny-finetuned-enron-spam-detection` | Binary spam/ham classification |
| 2 | `category_detection.py` | `google/gemma-2-9b-it` (Q4_K_M GGUF) | Two-step hierarchical classification |

### Two-Stage Classification Flow

1. **Spam Gate** — Fast BERT-tiny model filters spam emails. Spam emails return `(Spam, Spam)` immediately.
2. **Parent Category** — Gemma 2 9B classifies the email into one of 10 parent categories.
3. **Child Category** — Given the parent, Gemma 2 9B selects the most specific subcategory.

## Category Hierarchy

| Parent | Example Subcategories |
|--------|----------------------|
| **Work** | Meetings, Projects, Payroll, Clients, Recruitment, Schedules, Memos, Approvals, Training, Reviews, Expenses, Contracts, Strategy, Equipment, Onboarding, Operations, Others |
| **Finance** | Banking, Investments, Taxes, Bills, Loans, Insurance, Crypto, Mortgages, Grants, Others |
| **Purchases** | Orders, Shipping, Subscriptions, Refunds, Groceries, Services, Receipts, Invoices, Memberships, Others |
| **Travel** | Flights, Hotels, Transit, Itineraries, Visas, Rewards, Trains, Cruises, Rentals, Others |
| **Education** | Classes, Assignments, Notices, Exams, Certifications, Grades, Alumni, Scholarships, Admissions, Others |
| **Security** | Verification, Passwords, Alerts, Privacy, Fraud, Devices, Backups, Breaches, Compliance, Others |
| **Social** | Family, Friends, Platforms, Events, Groups, Dating, Networking, Forums, Messages, Others |
| **Promotions** | Newsletters, Sales, Updates, Surveys, Charity, Giveaways, Coupons, Campaigns, Partnerships, Others |
| **Spam** | Phishing, Scams, Junk, Fakes, Malware, Clickbait, Spoofing, Blackmail, Others |
| **Personal** | Health, Fitness, Legal, Housing, Pets, Hobbies, Civic, Medical, Auto, Others |

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| ML Frameworks | PyTorch, HuggingFace Transformers |
| LLM Inference | llama.cpp (GGUF), CPU-optimized |
| Utilities | scikit-learn, sentence-transformers |
| Environment | python-dotenv |

### Key Dependencies

- `transformers` — HuggingFace pipelines for BERT-based spam detection
- `torch` — Backend for deep learning models
- `llama-cpp-python` — CPU inference engine for Gemma 2 9B
- `huggingface-hub` — Download model weights from HuggingFace
- `python-dotenv` — Environment variable management
- `scikit-learn`, `sentence-transformers` — ML utilities

### Hardware Requirements

- **RAM**: 16GB (recommended for Gemma 2 9B Q4_K_M ~5.6GB)
- **Storage**: ~6GB for model weights
- **GPU**: Optional (auto-detected; CPU fallback with llama.cpp)
- **CPU**: 4+ cores recommended

## Project Structure

```
Invox/
├── .env                          # HuggingFace token (HF_TOKEN)
├── README.md
├── requirements.txt
├── generate_emails.py            # Dataset generator
├── update_emails.py              # Dataset post-processor
├── HelpingNotes/
│   ├── Main.md                   # Working state notes
│   └── report.md                 # Detailed analysis report
└── services/
    └── invox-ai/
        ├── pyproject.toml         # Package config & dependencies
        ├── src/invox/
        │   ├── __init__.py
        │   ├── api/               # API placeholder
        │   ├── features/
        │   │   └── email_classification/
        │   │       ├── __init__.py
        │   │       ├── service.py              # Main pipeline entry point
        │   │       ├── categories.py            # Category hierarchy definitions
        │   │       ├── spam_detection.py        # Stage 1: Spam filter
        │   │       └── category_detection.py    # Stage 2: Hierarchical classifier
        └── tests/
            └── features/email_classification/
                ├── test.sh                     # Benchmark runner
                ├── generate_dataset.py          # Synthetic email generator
                └── email_dataset/
                    ├── answer.txt               # Ground truth labels
                    └── 001.txt - 212.txt        # Test emails
```

## Usage

### Single Email Classification

```bash
cd services/invox-ai
export PYTHONPATH="$PWD/src"
python src/invox/features/email_classification/service.py email.txt
```

Output format:
```
RESULT|<filename>|<parent>|<subcategory>|<time_seconds>
```

### Benchmarking

Run the full benchmark against the 200-email test dataset:

```bash
cd services/invox-ai/tests/features/email_classification
./test.sh          # Run all 212 emails
./test.sh -10      # Run 10 random emails
./test.sh -all     # Explicitly run all
```

Metrics reported:
- **Full Pass** — Both parent and subcategory match (100%)
- **Partial Pass** — Only parent matches (50%)
- **Full Fail** — Neither matches (0%)
- **Avg Time** — Average inference time per email

### Dataset Generation

Generate synthetic email test data:

```bash
cd services/invox-ai/tests/features/email_classification
python generate_dataset.py
```

Creates 200 numbered `.txt` email files and an `answer.txt` with ground truth labels.

### Error Handling

| Condition | Output |
|-----------|--------|
| No input file | `RESULT|Error|NoInputFile|0.00` |
| File not found | `RESULT|<file>|Error|FileNotFound|0.00` |
| Runtime failure | `RESULT|<file>|Error|RuntimeFailure|0.00` |

## Setup

1. **Clone the repository**

2. **Create a Python virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   The actual dependencies are defined in `services/invox-ai/pyproject.toml`. Install them with:

   ```bash
   cd services/invox-ai
   pip install -e .
   ```

   Additionally, install runtime dependencies not yet in `pyproject.toml`:

   ```bash
   pip install llama-cpp-python huggingface-hub python-dotenv
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```
   HF_TOKEN=<your_huggingface_token>
   ```

5. **Download models** (automatic on first run)

   - BERT-tiny spam model (~100MB) — downloaded via HuggingFace pipeline
   - Gemma 2 9B Q4_K_M GGUF (~5.6GB) — downloaded via `huggingface-hub`

## License

Internal project — Invox Team
