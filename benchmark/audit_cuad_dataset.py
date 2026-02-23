"""Analisi del dataset CUAD: quanti item hanno clausola presente vs assente."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from datasets import load_dataset
from benchmarks.cuad import CATEGORY_MATCHERS, _match_category

ds = load_dataset("alex-apostolo/filtered-cuad", split="test", verification_mode="no_checks")

total = 0
empty_answers = 0
by_category = {}

for row in ds:
    question = row.get("question", "")
    cat = _match_category(question)
    if cat is None:
        continue
    raw = row.get("answers", {})
    if isinstance(raw, dict):
        texts = raw.get("text", [])
    elif isinstance(raw, list):
        texts = raw
    else:
        texts = []
    answers = [a for a in texts if isinstance(a, str) and a.strip()]
    total += 1
    has_clause = len(answers) > 0
    if not has_clause:
        empty_answers += 1
    if cat not in by_category:
        by_category[cat] = {"total": 0, "empty": 0}
    by_category[cat]["total"] += 1
    by_category[cat]["empty"] += (0 if has_clause else 1)

print(f"Dataset totale item IP: {total}")
print(f"Clausola PRESENTE: {total - empty_answers} ({(total-empty_answers)/total:.1%})")
print(f"Clausola ASSENTE:  {empty_answers} ({empty_answers/total:.1%})")
print()
print(f"{'Categoria':<25} {'Totale':>7} {'Presente':>9} {'Assente':>8}")
print("-" * 52)
for cat, v in sorted(by_category.items()):
    pres = v["total"] - v["empty"]
    print(f"{cat:<25} {v['total']:>7} {pres:>9} {v['empty']:>8}")
print()
print(f"=> Su 80 item campionati (10 per cat), circa {int(80*(total-empty_answers)/total)} hanno clausola presente")
print(f"=> Circa {int(80*empty_answers/total)} hanno clausola assente")
