# 📊 Invox Project - Detailed Analysis Report

**Version:** 1.0  
**Date:** 2026-02-25  
**Author:** Project Analysis

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Features & Control Flow](#4-features--control-flow)
5. [Execution Modes](#5-execution-modes)
6. [Future Roadmap](#6-future-roadmap)
7. [Configuration](#7-configuration)
8. [Summary](#8-summary)

---

## 1. Project Overview

### 1.1 What is Invox?

**Invox** is an AI services platform focused on **email classification** using Natural Language Processing (NLP). The project aims to automatically categorize incoming emails into predefined categories to streamline workflows and improve efficiency.

### 1.2 Primary Objective

- **Main Goal:** Provide AI services for advanced email classification
- **Problem Solved:** Automating the categorization of incoming emails to enable automated processing or routing based on content

### 1.3 Sub-Goals / Milestones

- [x] Develop and integrate multiple zero-shot classification models
- [x] Establish a robust benchmarking framework
- [x] Define comprehensive email categories
- [x] Implement core classification service with NLP models

---

## 2. Tech Stack

### 2.1 Core Technologies

| Category           | Technology                          | Purpose                         |
| ------------------ | ----------------------------------- | ------------------------------- |
| **Language**       | Python 3.9+                         | Primary programming language    |
| **ML Framework**   | PyTorch, Transformers               | Deep learning & NLP             |
| **LLM Inference**  | llama.cpp (GGUF)                    | CPU-optimized LLM execution     |
| **Classification** | scikit-learn, sentence-transformers | ML utilities & embeddings       |
| **Model Format**   | GGUF (Quantized)                    | Compressed LLM for CPU          |
| **Environment**    | Python venv, dotenv                 | Dependency & secrets management |

### 2.2 Key Dependencies

```
transformers>=4.0
torch>=2.0
scikit-learn>=1.0
sentence-transformers
llama-cpp-python
huggingface-hub
python-dotenv
```

### 2.3 Hardware Requirements

| Component   | Specification               |
| ----------- | --------------------------- |
| **RAM**     | 16GB (for GGUF model)       |
| **Storage** | 6GB+ (for models)           |
| **GPU**     | Optional (auto-detected)    |
| **CPU**     | Ryzen 5 (4 cores optimized) |

---

## 3. Project Structure

### 3.1 Directory Tree

```
Invox/
├── .env                                   # Environment variables (HF_TOKEN)
├── .gitignore                             # Git ignore rules
├── README.md                              # Project documentation
├── requirements.txt                       # Root dependencies (empty)
├── HelpingNotes/
│   └── Main.md                           # Project working notes
├── experiments/
│   └── benchmarks/
│       └── benchmark_all_models.sh       # Quick benchmark script
└── services/
    └── invox-ai/                         # Main service
        ├── pyproject.toml                 # Package configuration
        ├── src/invox/
        │   ├── __init__.py
        │   ├── api/
        │   │   └── __init__.py           # API modules (placeholder)
        │   ├── classification/
        │   │   └── __init__.py          # Classification modules (placeholder)
        │   └── features/
        │       └── email_classification/
        │           ├── __init__.py
        │           ├── service.py        # Main entry point
        │           ├── category_detection.py  # Gemma 2 LLM classifier
        │           ├── spam_detection.py      # BERT spam filter
        │           └── categories.py          # Category definitions
        └── tests/features/email_classification/
            ├── test.sh                   # Full benchmark script
            ├── generate_dataset.py       # Test data generator
            └── email_dataset/
                ├── 001.txt - 200.txt    # 200 test emails
                └── answer.txt            # Ground truth labels
```

### 3.2 File Descriptions

| File                      | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `.env`                    | Contains HuggingFace API token                 |
| `service.py`              | Main classification pipeline                   |
| `category_detection.py`   | Hierarchical category classifier using Gemma 2 |
| `spam_detection.py`       | Binary spam/ham classifier                     |
| `categories.py`           | Category hierarchy definitions                 |
| `test.sh`                 | Comprehensive benchmark script                 |
| `generate_dataset.py`     | Synthetic email generator                      |
| `benchmark_all_models.sh` | Quick model comparison                         |

---

## 4. Features & Control Flow

### 4.1 Feature Overview

The project implements a **two-stage email classification pipeline**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMAIL CLASSIFICATION PIPELINE                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   EMAIL INPUT   │
                    │  (Text Content) │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      STAGE 1: SPAM GATE       │
              │  ┌────────────────────────┐  │
              │  │ BERT-tiny Model        │  │
              │  │ (mrm8488/bert-tiny-   │  │
              │  │  finetuned-enron-     │  │
              │  │  spam-detection)      │  │
              │  └────────────────────────┘  │
              └─────────────┬────────────────┘
                            │
                     ┌──────┴──────┐
                     │  is_spam?   │
                     └──────┬──────┘
              ┌─────────────┴─────────────┐
              │                           │
             YES                          NO
              │                           │
              ▼                           ▼
    ┌─────────────────┐        ┌──────────────────────────┐
    │  RETURN SPAM   │        │  STAGE 2: CATEGORY       │
    │  cat: Spam     │        │  DETECTION               │
    │  subcat: Spam  │        │                          │
    └─────────────────┘        │  ┌────────────────────┐  │
                               │  │ Gemma 2 9B (GGUF) │  │
                               │  │ - Step 1: Parent  │  │
                               │  │ - Step 2: Child   │  │
                               │  └────────────────────┘  │
                               └────────────┬─────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │  FINAL OUTPUT       │
                                  │  cat: <Parent>      │
                                  │  subcat: <Child>   │
                                  └─────────────────────┘
```

---

### 4.2 Stage 1: Spam Detection

#### File: `spam_detection.py`

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Model**          | `mrm8488/bert-tiny-finetuned-enron-spam-detection` |
| **Task**           | Binary classification (Spam vs Ham)                |
| **Implementation** | Hugging Face pipeline                              |
| **Device**         | Auto (GPU if available)                            |
| **Max Length**     | 512 tokens                                         |
| **Output**         | Boolean (`True` if spam)                           |

#### Code Flow

```python
# 1. Initialize pipeline on module load
spam_classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-enron-spam-detection",
    device_map="auto",
    truncation=True,
    max_length=512,
)

# 2. Check spam on demand
def check_is_spam(email_text: str) -> bool:
    result = spam_classifier(email_text)[0]
    label = result["label"].lower()
    score = result["score"]
    return label == "spam" or label == "label_1"
```

---

### 4.3 Stage 2: Category Detection

#### File: `category_detection.py`

| Property             | Value                                |
| -------------------- | ------------------------------------ |
| **Model**            | `google/gemma-2-9b-it`               |
| **Quantization**     | Q4_K_M GGUF (~5.6GB)                 |
| **Inference Engine** | llama.cpp                            |
| **Context Window**   | 2048 tokens                          |
| **Temperature**      | 0.0 (deterministic)                  |
| **Max Tokens**       | 15                                   |
| **Approach**         | Two-step hierarchical classification |

#### Hierarchical Classification Flow

```
┌────────────────────────────────────────────────────────────┐
│              STEP 1: PARENT CATEGORY                       │
│                                                            │
│  Input: Email text (first 1500 chars)                     │
│  Prompt: "Classify into ONE of these categories:          │
│           [Work, Finance, Purchases, Travel, Education,    │
│            Security, Social, Promotions, Spam, Personal]" │
│  Output: Parent Category (e.g., "Work")                    │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│              STEP 2: CHILD CATEGORY                        │
│                                                            │
│  Input: Same email text + Parent Category                  │
│  Prompt: "Pick the most specific sub-category from:       │
│           [Meetings, Projects, Payroll, Clients, ...]"     │
│  Output: Child Category (e.g., "Meetings")                 │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  (Parent, Child)  │
              │  ("Work",         │
              │   "Meetings")     │
              └───────────────────┘
```

---

### 4.4 Category Hierarchy

#### Parent Categories (10)

| #   | Category   | Description                      |
| --- | ---------- | -------------------------------- |
| 1   | Work       | Professional/work-related emails |
| 2   | Finance    | Banking, investments, taxes      |
| 3   | Purchases  | Orders, shipping, subscriptions  |
| 4   | Travel     | Flights, hotels, itineraries     |
| 5   | Education  | Classes, exams, grades           |
| 6   | Security   | Verification, passwords, alerts  |
| 7   | Social     | Family, friends, events          |
| 8   | Promotions | Sales, newsletters, offers       |
| 9   | Spam       | Phishing, scams, junk            |
| 10  | Personal   | Health, fitness, legal           |

#### Subcategories Detail

| Parent         | Subcategories                                                                                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Work**       | Meetings, Projects, Payroll, Clients, Recruitment, Schedules, Memos, Approvals, Training, Reviews, Expenses, Contracts, Strategy, Equipment, Onboarding, Operations, Others |
| **Finance**    | Banking, Investments, Taxes, Bills, Loans, Insurance, Crypto, Mortgages, Grants, Others                                                                                     |
| **Purchases**  | Orders, Shipping, Subscriptions, Refunds, Groceries, Services, Receipts, Invoices, Memberships, Others                                                                      |
| **Travel**     | Flights, Hotels, Transit, Itineraries, Visas, Rewards, Trains, Cruises, Rentals, Others                                                                                     |
| **Education**  | Classes, Assignments, Notices, Exams, Certifications, Grades, Alumni, Scholarships, Admissions, Others                                                                      |
| **Security**   | Verification, Passwords, Alerts, Privacy, Fraud, Devices, Backups, Breaches, Compliance, Others                                                                             |
| **Social**     | Family, Friends, Platforms, Events, Groups, Dating, Networking, Forums, Messages, Others                                                                                    |
| **Promotions** | Newsletters, Sales, Updates, Surveys, Charity, Giveaways, Coupons, Campaigns, Partnerships, Others                                                                          |
| **Spam**       | Phishing, Scams, Junk, Fakes, Malware, Clickbait, Spoofing, Blackmail, Others                                                                                               |
| **Personal**   | Health, Fitness, Legal, Housing, Pets, Hobbies, Civic, Medical, Auto, Others                                                                                                |

---

## 5. Execution Modes

### 5.1 CLI Mode

#### Usage

```bash
python service.py <email_file.txt>
```

#### Example

```bash
# Run classification on a single email
python services/invox-ai/src/invox/features/email_classification/service.py email.txt

# Output:
# cat: Work
# subcat: Meetings
```

#### Output Format

```
cat: <parent_category>
subcat: <child_category>
```

#### Error States

| Error          | Output                               |
| -------------- | ------------------------------------ |
| No input file  | `cat: Error\nsubcat: NoInputFile`    |
| File not found | `cat: Error\nsubcat: FileNotFound`   |
| Import failure | `cat: Error\nsubcat: ImportFailure`  |
| Runtime error  | `cat: Error\nsubcat: RuntimeFailure` |

---

### 5.2 Benchmark Mode

#### Usage

```bash
# Run all 200 emails
./test.sh

# Run N random emails
./test.sh -N
```

#### Metrics Calculated

| Metric           | Description                                |
| ---------------- | ------------------------------------------ |
| **Full Pass**    | Both parent AND child correct (100% match) |
| **Partial Pass** | Only parent correct (50% match)            |
| **Full Fail**    | Neither correct (0% match)                 |
| **Avg Time**     | Average execution time per email           |

#### Benchmark Output

```
+-----------+------------------------+-----------+------------------------+--------+
| Email     | Prediction             | Time      | Actual                 | Match  |
+-----------+------------------------+-----------+------------------------+--------+
| 001.txt   | Work,Meetings          | 2.34s     | Work,Meetings         | 100    |
| 002.txt   | Finance,Banking        | 2.45s     | Finance,Investments   | 50     |
| 003.txt   | Spam,Phishing          | 1.12s     | Spam,Phishing          | 100    |
+-----------+------------------------+-----------+------------------------+--------+

============= FINAL METRICS =============

Full Pass   : 150
Partial Pass: 30
Full Fail   : 20
Total Runs  : 200
Avg Time    : 2.15 sec
```

---

### 5.3 Dataset Generation Mode

#### Usage

```bash
python generate_dataset.py
```

#### Output

- `email_dataset/001.txt` - `email_dataset/200.txt` (200 synthetic emails)
- `email_dataset/answer.txt` (ground truth labels)

#### Generation Process

```
┌─────────────────────────────────────────┐
│         DATASET GENERATOR              │
├─────────────────────────────────────────┤
│ 1. Round-robin category selection      │
│ 2. Template-based content generation   │
│ 3. Faker data injection               │
│ 4. Word count padding (180-350 words)  │
│ 5. Save to numbered .txt files        │
└─────────────────────────────────────────┘
```

---

## 6. Future Roadmap

### 6.1 Optimization & Deployment Plan

```
┌─────────────────────────────────────────────────────────────────┐
│                  FUTURE OPTIMIZATION ROADMAP                   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  1. FINE-TUNING│    │ 2. GPU SERVER │    │ 3. CONTINUOUS │
│      (LoRA)   │    │   DEPLOYMENT  │    │    BATCHING   │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Train LoRA    │    │ Replace       │    │ Process 50+   │
│ adapter on    │    │ llama.cpp     │    │ emails at     │
│ gemma-2-9b-it │    │ with vLLM     │    │ once via API  │
│               │    │               │    │               │
│ - Edge cases  │    │ - GPU optimized│    │ - Higher      │
│ - Specific    │    │ - Max throughput│  │   throughput  │
│   domain      │    │               │    │ - Parallel    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
              ┌───────────────────────────────┐
              │   MERGED DEPLOYMENT           │
              │   - Custom .gguf export       │
              │   - OR vLLM with quantization │
              └───────────────────────────────┘
```

### 6.2 GPU Deployment Options

| Option   | VRAM Required | Precision | Use Case         |
| -------- | ------------- | --------- | ---------------- |
| bfloat16 | ~24GB         | Native    | Maximum accuracy |
| AWQ      | ~12GB         | Quantized | GPU-optimized    |
| FP8      | ~12GB         | Quantized | Fast inference   |

---

## 7. Configuration

### 7.1 Environment Variables

#### `.env` File

### 7.2 Model Download

Models are downloaded automatically on first run:

| Model                          | Size   | Location          |
| ------------------------------ | ------ | ----------------- |
| bert-tiny-finetuned-enron-spam | ~100MB | HuggingFace cache |
| gemma-2-9b-it-Q4_K_M.gguf      | ~5.6GB | HuggingFace Hub   |

### 7.3 Python Configuration

#### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "invox-ai"
version = "0.1.0"
description = "AI services for Invox"
authors = [{name = "Invox Team", email = "dev@invox.com"}]
dependencies = [
    "transformers",
    "torch",
    "scikit-learn",
    "sentence-transformers"
]
requires-python = ">=3.9"

[tool.setuptools.packages.find]
where = ["src"]
```

---

## 8. Summary

### 8.1 Project At A Glance

| Aspect             | Details                                |
| ------------------ | -------------------------------------- |
| **Project Type**   | AI Email Classification Service        |
| **Primary Use**    | Automated email routing/categorization |
| **Architecture**   | Two-stage pipeline (Spam → Category)   |
| **LLM Backend**    | Gemma 2 9B (CPU) + BERT-tiny (Spam)    |
| **Classification** | Hierarchical zero-shot classification  |
| **Categories**     | 10 parent + ~100 child categories      |
| **Test Data**      | 200 synthetic emails                   |
| **Benchmarking**   | Accuracy + performance metrics         |

### 8.2 Key Files Reference

| File Path                                                                         | Purpose                   |
| --------------------------------------------------------------------------------- | ------------------------- |
| `services/invox-ai/src/invox/features/email_classification/service.py`            | Main pipeline entry point |
| `services/invox-ai/src/invox/features/email_classification/spam_detection.py`     | Spam filter module        |
| `services/invox-ai/src/invox/features/email_classification/category_detection.py` | Category classifier       |
| `services/invox-ai/src/invox/features/email_classification/categories.py`         | Category definitions      |
| `services/invox-ai/tests/features/email_classification/test.sh`                   | Benchmark script          |
| `services/invox-ai/tests/features/email_classification/generate_dataset.py`       | Test data generator       |

### 8.3 Control Flow Summary

```
┌────────────────────────────────────────────────────────────────┐
│                      CONTROL FLOW SUMMARY                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  START → Load Environment & Models                            │
│            │                                                   │
│            ▼                                                   │
│  Read Email File                                               │
│            │                                                   │
│            ▼                                                   │
│  ┌─────────────────┐                                           │
│  │ SPAM DETECTION │ ◄── BERT-tiny (fast, lightweight)         │
│  │ (spam_detection)│                                            │
│  └────────┬────────┘                                           │
│           │                                                    │
│      is_spam?                                                   │
│           │                                                    │
│    ┌──────┴──────┐                                             │
│    │             │                                             │
│   YES            NO                                            │
│    │             │                                             │
│    ▼             ▼                                             │
│  SPAM       ┌─────────────────────┐                            │
│  RESULT     │ CATEGORY DETECTION │ ◄── Gemma 2 9B (LLM)        │
│             │ (category_detection)│                            │
│             │ 1. Parent Step     │                             │
│             │ 2. Child Step      │                             │
│             └──────────┬──────────┘                            │
│                        │                                       │
│                        ▼                                       │
│              FINAL OUTPUT                                      │
│              "cat: X\nsubcat: Y"                              │
│                        │                                       │
│                        ▼                                       │
│                    END                                          │
└────────────────────────────────────────────────────────────────┘
```

---

### 8.4 Technology Stack Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Python     │    │   PyTorch    │    │Transformers  │   │
│  │   3.9+       │    │   (Deep      │    │  (Hugging    │   │
│  │              │    │   Learning)  │    │   Face)      │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │            │
│         └───────────────────┼───────────────────┘            │
│                             ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    INFERENCE ENGINE                       │ │
│  │  ┌─────────────────────┐  ┌──────────────────────────┐  │ │
│  │  │   llama.cpp         │  │   HuggingFace Pipeline   │  │ │
│  │  │   (GGUF - CPU)      │  │   (BERT - GPU/CPU)        │  │ │
│  │  └─────────────────────┘  └──────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                             │                                 │
│                             ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                      MODELS                               │ │
│  │  ┌─────────────────────┐  ┌──────────────────────────┐   │ │
│  │  │ Gemma 2 9B         │  │ bert-tiny-finetuned      │   │ │
│  │  │ (Category Classif) │  │ Enron-Spam-Detection    │   │ │
│  │  └─────────────────────┘  └──────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Sample Email Formats

### Work - Meetings

```
From: "John Smith" <john.smith@company.com>
To: "Jane Doe" <jane.doe@example.com>
Date: Mon, 15 Jan 2024 09:00:00 +0000
Subject: Meeting Reminder: Q3 Planning

Hi Team,

Just a reminder about our Q3 Planning meeting scheduled for Friday.
We will be discussing the Q3 roadmap and budget allocation.

Best,
John Smith

====================
CONFIDENTIALITY NOTICE: The contents of this email message...
```

### Spam - Phishing

```
From: "Admin" <admin@secure-account.net>
To: "User" <user@example.com>
Date: Mon, 15 Jan 2024 02:00:00 +0000
Subject: URGENT: Account Suspended!

Dear User,

Your account has been flagged for suspicious activity.
Click here to verify your identity or your account will be
deleted in 24 hours.

Admin

====================
CONFIDENTIALITY NOTICE: The contents of this email message...
```

---

## Appendix B: Error Handling

| Error Type       | Cause                    | Output                               |
| ---------------- | ------------------------ | ------------------------------------ |
| `NoInputFile`    | No CLI argument provided | `cat: Error\nsubcat: NoInputFile`    |
| `FileNotFound`   | File path doesn't exist  | `cat: Error\nsubcat: FileNotFound`   |
| `ImportFailure`  | Module import error      | `cat: Error\nsubcat: ImportFailure`  |
| `RuntimeFailure` | Any runtime exception    | `cat: Error\nsubcat: RuntimeFailure` |

---

**End of Report**

---

_Generated: 2026-02-25_  
_Project: Invox AI Services_  
_Version: 1.0_
