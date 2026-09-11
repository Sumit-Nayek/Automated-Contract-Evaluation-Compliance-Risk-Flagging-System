"""
Month 2 - Step 1: Sliding-Window Chunked QA Evaluation.

Directly addresses the Month-1 finding: 56.6% of answerable questions
scored F1=0.0 because the gold clause was truncated out of the context
window. Instead of passing the first N characters, this splits each
contract into overlapping windows sized to the model's max sequence
length, runs QA on every window, and keeps the highest-confidence answer.

Reuses the scoring functions from baseline_eval.py so results are
directly comparable to the Month-1 number.
"""

from pathlib import Path
import json

from baseline_eval import (
    N_PER_CATEGORY,
    RANDOM_SEED,
    OUTPUT_DIR,
    stratified_sample,
    best_score_over_golds,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# distilbert-base-cased-distilled-squad has a 512-token max sequence length.
# Reserve room for the question + special tokens; ~1 token ~= 4 chars for
# English legal text, so this is a conservative character-based window.
WINDOW_CHARS = 1600
STRIDE_CHARS = 400   # overlap so an answer straddling a window boundary
                      # still appears intact in at least one window
NO_ANSWER_CONF_THRESHOLD = 0.15


def chunk_context(context: str, window_chars: int, stride_chars: int):
    """Yield overlapping (start_offset, window_text) pairs covering context."""
    if len(context) <= window_chars:
        yield 0, context
        return
    step = window_chars - stride_chars
    for start in range(0, len(context), step):
        window = context[start:start + window_chars]
        if not window:
            break
        yield start, window
        if start + window_chars >= len(context):
            break


def run_chunked_baseline():
    from datasets import load_dataset
    import pandas as pd
    from transformers import pipeline

    print("Loading CUAD dataset...")
    dataset = load_dataset("theatticusproject/cuad-qa")
    train_df = pd.DataFrame(dataset["train"])

    print(f"Sampling {N_PER_CATEGORY} example(s) per category...")
    sample_df = stratified_sample(train_df, N_PER_CATEGORY, RANDOM_SEED)
    print(f"Total evaluation examples: {len(sample_df)}")

    print("Loading zero-shot QA model...")
    qa_pipeline = pipeline(
        "question-answering", model="distilbert-base-cased-distilled-squad"
    )

    per_example_results = []

    for idx, row in sample_df.iterrows():
        context = row["context"]
        question = row["question"]
        gold_texts = row["answers"]["text"] if row["answers"]["text"] else []
        is_answerable = len(gold_texts) > 0

        windows = list(chunk_context(context, WINDOW_CHARS, STRIDE_CHARS))
        best = {"answer": "", "score": -1.0}
        for _, window_text in windows:
            result = qa_pipeline(question=question, context=window_text)
            if result["score"] > best["score"]:
                best = result

        record = {
            "category": row["category"],
            "is_answerable": is_answerable,
            "n_windows": len(windows),
            "pred_answer": best["answer"],
            "pred_confidence": round(best["score"], 4),
        }

        if is_answerable:
            f1, em = best_score_over_golds(best["answer"], gold_texts)
            record["f1"] = round(f1, 4)
            record["em"] = em
            record["hallucinated_on_no_answer"] = None
        else:
            record["f1"] = None
            record["em"] = None
            record["hallucinated_on_no_answer"] = best["score"] >= NO_ANSWER_CONF_THRESHOLD

        per_example_results.append(record)

        if (idx + 1) % 20 == 0:
            print(f"  ...{idx + 1}/{len(sample_df)} done")

    df_results = pd.DataFrame(per_example_results)

    answerable = df_results[df_results["is_answerable"]]
    unanswerable = df_results[~df_results["is_answerable"]]

    summary = {
        "model": "distilbert-base-cased-distilled-squad (zero-shot, chunked)",
        "window_chars": WINDOW_CHARS,
        "stride_chars": STRIDE_CHARS,
        "n_examples_total": len(df_results),
        "n_answerable": int(len(answerable)),
        "n_unanswerable": int(len(unanswerable)),
        "overall_f1": round(float(answerable["f1"].mean()), 4) if len(answerable) else None,
        "overall_em": round(float(answerable["em"].mean()), 4) if len(answerable) else None,
        "hallucination_rate_on_unanswerable": (
            round(float(unanswerable["hallucinated_on_no_answer"].mean()), 4)
            if len(unanswerable) else None
        ),
        "avg_windows_per_contract": round(float(df_results["n_windows"].mean()), 2),
    }

    df_results.to_csv(OUTPUT_DIR / "chunked_baseline_per_example.csv", index=False)
    with open(OUTPUT_DIR / "chunked_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Chunked Baseline Summary (compare to Month 1's 0.0346 F1) ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary, df_results


if __name__ == "__main__":
    run_chunked_baseline()