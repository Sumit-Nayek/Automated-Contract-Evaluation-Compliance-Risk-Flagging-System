"""
Month 1 - Step: Quantified Zero-Shot Baseline Evaluation on CUAD.

Turns the single-example sanity check into a real, loggable baseline:
- Samples N questions per clause category (all 41 CUAD categories)
- Runs the zero-shot QA pipeline (distilbert-base-cased-distilled-squad)
- Scores answerable questions with SQuAD-style token F1 / Exact Match
- Scores unanswerable questions (no clause of that type present) by whether
  the model correctly abstains (below a confidence threshold) or hallucinates
  an answer anyway
- Saves per-category and overall results to CSV so Month 2's fine-tuned
  model has a documented number to beat

Run inside your Kaggle/Colab notebook (same environment as Month 1 cells),
or as a standalone script: `python baseline_eval.py`
"""

import re
import string
import collections
import random
import json
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_PER_CATEGORY = 4          # keep total calls modest for free-tier compute
CONTEXT_CHAR_LIMIT = 2500   # same truncation as the Month-1 notebook baseline
NO_ANSWER_CONF_THRESHOLD = 0.15   # below this, we treat the model as "abstaining"
RANDOM_SEED = 42
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# SQuAD-style normalization + F1 / EM (standard implementation)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def best_score_over_golds(pred: str, gold_texts):
    """CUAD answers can have multiple valid gold spans; take the max score."""
    if not gold_texts:
        return 0.0, 0
    f1 = max(compute_f1(pred, g) for g in gold_texts)
    em = max(compute_em(pred, g) for g in gold_texts)
    return f1, em


# ---------------------------------------------------------------------------
# CUAD category extraction
# ---------------------------------------------------------------------------
def get_category(example) -> str:
    """
    CUAD's HF 'id' field is formatted as
    '<contract_title>__<Category Name>_<qa_index>', e.g.
    'LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Document Name_0'.
    Strip the trailing '_<index>' so all QA pairs for the same clause type
    collapse into one category. Fall back to the raw question text if the
    id doesn't split cleanly, so the script never silently drops an example.
    """
    ex_id = example.get("id", "")
    if "__" in ex_id:
        category = ex_id.split("__")[-1]
        category = re.sub(r"_\d+$", "", category)
        return category
    return example.get("question", "UNKNOWN_CATEGORY")[:60]


# ---------------------------------------------------------------------------
# Sampling: N examples per category, stratified, reproducible
# ---------------------------------------------------------------------------
def stratified_sample(train_df, n_per_category: int, seed: int):
    random.seed(seed)
    train_df = train_df.copy()
    train_df["category"] = train_df.apply(get_category, axis=1)

    sampled_rows = []
    for category, group in train_df.groupby("category"):
        take = min(n_per_category, len(group))
        sampled_rows.append(group.sample(n=take, random_state=seed))

    import pandas as pd
    return pd.concat(sampled_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_baseline():
    from datasets import load_dataset
    import pandas as pd
    from transformers import pipeline

    print("Loading CUAD dataset...")
    # "cuad" (no namespace) no longer resolves as a loadable dataset under
    # datasets==2.18.0 - use the maintained repo under its full path instead.
    dataset = load_dataset("theatticusproject/cuad-qa")
    train_df = pd.DataFrame(dataset["train"])

    print(f"Sampling {N_PER_CATEGORY} example(s) per category...")
    sample_df = stratified_sample(train_df, N_PER_CATEGORY, RANDOM_SEED)
    print(f"Total evaluation examples: {len(sample_df)} "
          f"across {sample_df['category'].nunique()} categories")

    print("Loading zero-shot QA baseline model...")
    qa_pipeline = pipeline(
        "question-answering", model="distilbert-base-cased-distilled-squad"
    )

    per_example_results = []

    for _, row in sample_df.iterrows():
        context = row["context"][:CONTEXT_CHAR_LIMIT]
        question = row["question"]
        gold_texts = row["answers"]["text"] if row["answers"]["text"] else []
        is_answerable = len(gold_texts) > 0

        result = qa_pipeline(question=question, context=context)
        pred_text = result["answer"]
        pred_score = result["score"]

        record = {
            "category": row["category"],
            "is_answerable": is_answerable,
            "pred_answer": pred_text,
            "pred_confidence": round(pred_score, 4),
        }

        if is_answerable:
            f1, em = best_score_over_golds(pred_text, gold_texts)
            record["f1"] = round(f1, 4)
            record["em"] = em
            record["hallucinated_on_no_answer"] = None
        else:
            # Correct behaviour = low confidence ("no answer" in this contract)
            hallucinated = pred_score >= NO_ANSWER_CONF_THRESHOLD
            record["f1"] = None
            record["em"] = None
            record["hallucinated_on_no_answer"] = hallucinated

        per_example_results.append(record)

    df_results = pd.DataFrame(per_example_results)

    # --- Aggregate: answerable questions ---
    answerable = df_results[df_results["is_answerable"]]
    overall_f1 = answerable["f1"].mean() if len(answerable) else float("nan")
    overall_em = answerable["em"].mean() if len(answerable) else float("nan")

    # --- Aggregate: unanswerable questions ---
    unanswerable = df_results[~df_results["is_answerable"]]
    hallucination_rate = (
        unanswerable["hallucinated_on_no_answer"].mean()
        if len(unanswerable) else float("nan")
    )

    per_category = (
        df_results[df_results["is_answerable"]]
        .groupby("category")[["f1", "em"]]
        .mean()
        .sort_values("f1")
    )

    # --- Save outputs ---
    df_results.to_csv(OUTPUT_DIR / "baseline_per_example.csv", index=False)
    per_category.to_csv(OUTPUT_DIR / "baseline_per_category.csv")

    summary = {
        "model": "distilbert-base-cased-distilled-squad (zero-shot)",
        "n_per_category": N_PER_CATEGORY,
        "context_char_limit": CONTEXT_CHAR_LIMIT,
        "n_examples_total": len(df_results),
        "n_answerable": int(len(answerable)),
        "n_unanswerable": int(len(unanswerable)),
        "overall_f1": round(float(overall_f1), 4),
        "overall_em": round(float(overall_em), 4),
        "hallucination_rate_on_unanswerable": round(float(hallucination_rate), 4),
    }
    with open(OUTPUT_DIR / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Zero-Shot Baseline Summary (Month 1 deliverable) ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {OUTPUT_DIR}/baseline_summary.json")
    print(f"Saved: {OUTPUT_DIR}/baseline_per_category.csv")
    print(f"Saved: {OUTPUT_DIR}/baseline_per_example.csv")

    return summary, per_category, df_results


if __name__ == "__main__":
    run_baseline()