"""
Diagnostic: inspect a handful of answerable examples to see WHY F1 is low.
Run after baseline_eval.py has produced results/baseline_per_example.csv.
"""
import pandas as pd

df = pd.read_csv("results/baseline_per_example.csv")
answerable = df[df["is_answerable"] == True].copy()

print(f"Answerable examples: {len(answerable)}")
print(f"F1 == 0.0 count: {(answerable['f1'] == 0.0).sum()} "
      f"({(answerable['f1'] == 0.0).mean():.1%})")
print()

# Show 5 worst-scoring examples: category, predicted answer, confidence, f1
worst = answerable.sort_values("f1").head(5)
for _, row in worst.iterrows():
    print(f"--- category: {row['category']} | f1: {row['f1']} ---")
    print(f"predicted: {row['pred_answer'][:200]!r}")
    print(f"confidence: {row['pred_confidence']}")
    print()