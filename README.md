# Automated Contract Evaluation & Risk Flagging System

An end-to-end NLP pipeline designed to ingest unstructured commercial contracts, extract 41 standard legal clauses using domain-adapted transformers (LEGAL-BERT), and flag protocol deviations using statistical similarity thresholds.

## Project Architecture (3-Month Timeline)
* **Phase 1 (Current):** Multi-format parsing (PyMuPDF) and baseline extraction over the CUAD v1 dataset.
* **Phase 2:** Fine-tuning transformer models for clause span extraction and developing the cosine-similarity risk threshold logic.
* **Phase 3:** Streamlit UI integration with real-time risk summaries and secure runtime API credential handling.

## Tech Stack
* **Environments:** Kaggle Notebooks
* **NLP & ML:** Hugging Face `transformers`, `datasets`, PyTorch, Scikit-learn
* **Data Processing:** Pandas, PyMuPDF