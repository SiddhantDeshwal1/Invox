# Current Working State Summary: Invox AI Services

## 1. Primary Objective
-   **Main goal:** To provide AI services, primarily focusing on advanced email classification using various Natural Language Processing (NLP) models.
-   **Problem solving:** Automating the categorization of incoming emails to streamline workflows, improve efficiency, and potentially enable further automated processing or routing based on content.

## 2. Sub-Goals / Milestones
-   Develop and integrate multiple zero-shot classification models for email categorization.
-   Establish a robust benchmarking framework to evaluate the performance (accuracy, speed) of different models against a defined email dataset.
-   Define a comprehensive set of email categories for classification.
-   Implement a core classification service utilizing a selected NLP model.

## 3. Decisions Made
-   **Technology Stack:** Python-based, leveraging `transformers`, `torch`, `scikit-learn`, and `sentence-transformers`.
-   **Core Classification Model:** The `google/flan-t5-large` model (implemented in `services/invox-ai/src/invox/features/email_classification/service.py`) is currently designated as the primary classification service.
-   **Benchmarking Approach:** Two distinct benchmarking scripts exist: one for quick performance checks (`benchmark_all_models.sh`) and another for detailed accuracy and performance metrics against a labeled dataset (`benchmark_models.sh`).
-   **Email Categories:** A fixed set of categories is defined in `invox/features/email_classification/categories.py`.

## 4. Assumptions
-   **Email Content:** Emails are primarily text-based, and classification is performed on the text content.
-   **Zero-shot Classification Efficacy:** The chosen zero-shot classification approach is suitable for the diverse range of email categories without requiring extensive labeled training data for each category.
-   **Model Availability:** Pre-trained models from Hugging Face Transformers are readily available and performant enough for the task.
-   **Hardware:** Availability of GPU (`torch.cuda.is_available()`) for accelerating model inference is assumed, with CPU as a fallback.

## 5. Constraints
-   **Model Token Limits:** Text truncation is applied to email content to fit within model token limits (e.g., T5's 512 tokens, with a character truncation to 2500 for safety; DeBERTa v3 to 4000 chars; GLiClass to 8000 chars).
-   **Classification Scope:** Only single-label classification (`multi_label=False`) is currently implemented for zero-shot models.
-   **Performance Requirements:** While not explicitly stated, the presence of benchmarking scripts implies an underlying requirement for reasonable inference speed and accuracy.
-   **Python Version:** Requires Python >= 3.9 (from `pyproject.toml`).

## 6. Key Definitions / Concepts Established
-   **Email Classification:** The automated process of assigning a predefined category to an email.
-   **Zero-shot Classification:** An NLP technique where a model classifies text into categories it hasn't been explicitly trained on, by leveraging its understanding of natural language inference.
-   **`LABELS`:** The predefined list of email categories used by all classification models.
-   **Hugging Face Transformers Pipeline:** A high-level API used for easy access to pre-trained models for various NLP tasks, specifically zero-shot classification.
-   **`CrossEncoder`:** A Sentence-Transformers model used in `nli_distilroberta_base.py` for ranking text pairs.

## 7. Important Data / Artifacts
-   **`README.md`**: Project overview.
-   **`requirements.txt`**: Python package dependencies for the overall project.
-   **`services/invox-ai/pyproject.toml`**: Project metadata and dependencies for the `invox-ai` service.
-   **`services/invox-ai/src/invox/features/email_classification/categories.py`**: Defines all possible email classification categories.
-   **`services/invox-ai/src/invox/features/email_classification/service.py`**: The main classification service, currently using `google/flan-t5-large`.
-   **`services/invox-ai/src/invox/models/experimental/*.py`**: A collection of Python scripts implementing various experimental zero-shot classification models (e.g., BART, DeBERTa, DistilBERT, Flan-T5, GLiClass, RoBERTa).
-   **`experiments/benchmarks/benchmark_all_models.sh`**: A shell script for quick benchmarking of all experimental models.
-   **`services/invox-ai/tests/features/models/benchmark_models.sh`**: A comprehensive benchmarking script for evaluating model accuracy and performance against a dataset.
-   **`services/invox-ai/tests/features/models/email_dataset/*.txt`**: A dataset of email text files for testing.
-   **`services/invox-ai/tests/features/models/email_dataset/answers.txt`**: Ground truth labels for the email dataset.

## 8. Open Questions / Unresolved Issues
-   **Model Selection for Production:** While `flan-t5-large` is in `service.py`, it's also in `experimental`. The criteria for promoting experimental models to production are not clear.
-   **Performance Optimization:** No explicit strategies for optimizing model inference speed or memory usage are detailed beyond leveraging GPU.
-   **Scalability:** How the service is intended to scale to handle a large volume of emails is not specified.
-   **Error Handling:** Basic file not found error handling exists, but more comprehensive error management for model loading/inference is not evident.
-   **Hypothesis Template for all models:** Not all zero-shot models explicitly use a `hypothesis_template` (e.g., `DeBERTa-v3`). Consistency might be beneficial.

## 9. Risks / Weak Points
-   **Zero-shot Accuracy Limitations:** Zero-shot models, while flexible, might not achieve the same accuracy as fine-tuned models for specific domains.
-   **Dependency Management:** The project uses `requirements.txt` at the root and `pyproject.toml` within `services/invox-ai`. This dual approach could lead to dependency conflicts or confusion if not managed carefully.
-   **Model Loading Time:** Loading large models can be time-consuming, as indicated by the benchmarking scripts. This could impact real-time classification performance.
-   **Hardcoded `device=-1` or `device_map="auto"`:** Relies on `transformers` to automatically handle device placement, which might not always be optimal or explicit.
-   **Text Truncation:** Truncating email text might lead to loss of critical information for classification in longer emails.

## 10. Current State Snapshot
-   The project is set up for developing and benchmarking email classification AI services.
-   A core service using Flan-T5 Large is identified, alongside several other experimental zero-shot models.
-   Comprehensive benchmarking capabilities are in place to evaluate model performance.
-   Email categories are well-defined.
-   The `HelpingNotes` directory and the `Main.md` file exist. This summary will overwrite the previous content of `Main.md`.