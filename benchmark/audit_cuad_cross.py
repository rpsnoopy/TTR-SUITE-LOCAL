"""
Analisi cross-model CUAD: per ogni posizione 0-79,
quanti modelli hanno F1=0 vs F1=1?
"""
import pandas as pd, glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = r"C:\TTR_Benchmark\results"
frames = []
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "raw_results_*.csv"))):
    frames.append(pd.read_csv(f))
df_all = pd.concat(frames, ignore_index=True)
if "task_id" in df_all.columns:
    df_all = df_all.drop_duplicates(subset=["model","benchmark","task_id"], keep="last")

cuad = df_all[df_all["benchmark"] == "cuad"].copy()
cuad["pos"] = cuad["task_id"].str.extract(r"::(\d+)$").astype(int)
cuad["ic_f"] = cuad["is_correct"].astype(float)

# 6 modelli completi
models_full = ["claude-sonnet-4-6","deepcoder-14b","gpt-oss-20b",
               "mistral-small-24b","qwen3-14b","qwen3-30b-a3b"]
cuad_f = cuad[cuad["model"].isin(models_full)]

# per ogni posizione: quanti modelli hanno F1=1 ?
pos_stats = cuad_f.groupby("pos")["ic_f"].agg(
    n_models="count",
    n_correct=lambda x: (x > 0).sum(),
    max_f1="max"
)

# posizioni dove NESSUN modello ha trovato la clausola
all_zero = pos_stats[pos_stats["n_correct"] == 0]
# posizioni dove TUTTI hanno trovato la clausola
all_found = pos_stats[pos_stats["n_correct"] == pos_stats["n_models"]]
# posizioni con split (alcuni trovano, altri no)
split_pos = pos_stats[(pos_stats["n_correct"] > 0) & (pos_stats["n_correct"] < pos_stats["n_models"])]

print(f"Posizioni CUAD 0-79 analizzate: 80")
print(f"Tutti trovano (F1=1 per tutti):  {len(all_found):2d}")
print(f"Nessuno trova (F1=0 per tutti):  {len(all_zero):2d}  <- probabilmente ground truth vuota o clausola assente")
print(f"Split (alcuni si, alcuni no):    {len(split_pos):2d}  <- clausola presente ma difficile")
print()

# campione delle posizioni dove nessuno trova - guardare le risposte
print("Campione posizioni all-zero (i modelli che estraggono qualcosa nonostante F1=0):")
for pos in sorted(all_zero.index)[:5]:
    rows = cuad_f[cuad_f["pos"] == pos][["model","response"]].head(3)
    print(f"\n  pos={pos}:")
    for _, row in rows.iterrows():
        resp = str(row["response"])[:80].replace("\n"," ")
        print(f"    {row['model']}: {resp!r}")

print()
print("Campione posizioni SPLIT (alcuni trovano, altri no):")
for pos in sorted(split_pos.index)[:3]:
    rows = cuad_f[cuad_f["pos"] == pos][["model","ic_f","response"]]
    print(f"\n  pos={pos}:")
    for _, row in rows.iterrows():
        resp = str(row["response"])[:80].replace("\n"," ")
        print(f"    {row['model']} F1={row['ic_f']:.2f}: {resp!r}")

# impatto sul ranking CUAD se escludiamo le posizioni all-zero (ground truth probabilmente vuota)
print()
print("=" * 70)
print("Impatto sul ranking CUAD se escludiamo le posizioni all-zero")
print("(assumendo che siano ground-truth vuote, cioe' valutazione corretta = 'nessuna')")
print("=" * 70)

non_zero_positions = set(all_found.index) | set(split_pos.index)
cuad_contested = cuad[cuad["pos"].isin(non_zero_positions)]
print(f"Item con clausola realmente presente: {len(non_zero_positions)}/80")
print()

# Ricalcola score solo su questi item
for model in models_full + ["phi4-14b"]:
    grp = cuad_contested[cuad_contested["model"] == model]
    if len(grp) == 0:
        continue
    ic = grp["is_correct"].astype(float)
    orig_grp = cuad[cuad["model"] == model]
    orig_score = orig_grp["is_correct"].astype(float).mean()
    new_score = ic.mean()
    print(f"  {model:<25}: orig={orig_score:.1%}  su-clauses-only={new_score:.1%}  (delta={new_score-orig_score:+.1%})")
