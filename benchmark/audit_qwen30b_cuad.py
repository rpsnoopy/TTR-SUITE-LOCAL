"""
Analisi dettagliata qwen3-30b-a3b CUAD: confronto run senza seed vs con seed.
"""
import pandas as pd, glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = r"C:\TTR_Benchmark\results"

# Carica tutti i CSV
frames = []
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "raw_results_*.csv"))):
    df = pd.read_csv(f)
    df["_src"] = os.path.basename(f)
    frames.append(df)
df_all = pd.concat(frames, ignore_index=True)

# NON dedup — vogliamo vedere TUTTI i run separatamente
q30_cuad = df_all[
    (df_all["model"] == "qwen3-30b-a3b") &
    (df_all["benchmark"] == "cuad")
].copy()

q30_cuad["pos"] = q30_cuad["task_id"].str.extract(r"::(\d+)$").astype(int)
q30_cuad["ic_f"] = q30_cuad["is_correct"].astype(float)
q30_cuad["resp_s"] = q30_cuad["response"].astype(str).str.strip()

# Mostra i run disponibili
print("Run disponibili per qwen3-30b-a3b CUAD:")
for src, grp in q30_cuad.groupby("_src"):
    score = grp["ic_f"].mean()
    n = len(grp)
    print(f"  {src}: n={n}, score={score:.1%}")

print()

# Prendi gli ultimi due run (no-seed e seed)
runs = q30_cuad["_src"].unique()
if len(runs) < 2:
    print("Solo un run disponibile, impossibile confrontare.")
    sys.exit(0)

run_prev = sorted(runs)[-2]  # run senza seed
run_new  = sorted(runs)[-1]  # run con seed

df_prev = q30_cuad[q30_cuad["_src"] == run_prev].set_index("pos")
df_new  = q30_cuad[q30_cuad["_src"] == run_new].set_index("pos")

print(f"Run PRECEDENTE (no seed): {run_prev}")
print(f"  Score: {df_prev['ic_f'].mean():.1%}, F1=1: {(df_prev['ic_f']==1).sum()}, F1=0: {(df_prev['ic_f']==0).sum()}")
print()
print(f"Run CORRENTE (seed=42): {run_new}")
print(f"  Score: {df_new['ic_f'].mean():.1%}, F1=1: {(df_new['ic_f']==1).sum()}, F1=0: {(df_new['ic_f']==0).sum()}")
print()

# Distribuzioni risposta
print("=" * 70)
print("Distribuzione risposte — run con seed=42")
print("=" * 70)
nessuna = df_new["resp_s"].str.lower().str.contains("nessuna|no clause|absent", na=False)
nan_resp = df_new["resp_s"].str.lower().isin(["nan", ""])
has_text = ~nessuna & ~nan_resp

print(f"  'NESSUNA' (no clause):  {nessuna.sum():3d}  ->  F1=1: {df_new[nessuna]['ic_f'].eq(1).sum()}, F1=0: {df_new[nessuna]['ic_f'].eq(0).sum()}")
print(f"  Estrazione testo:       {has_text.sum():3d}  ->  F1=1: {df_new[has_text]['ic_f'].eq(1).sum()}, F1=0: {df_new[has_text]['ic_f'].eq(0).sum()}")
print(f"  NaN/vuoto:              {nan_resp.sum():3d}  ->  F1=1: {df_new[nan_resp]['ic_f'].eq(1).sum()}, F1=0: {df_new[nan_resp]['ic_f'].eq(0).sum()}")
print()

# NaN responses nel dettaglio
if nan_resp.sum() > 0:
    print("=" * 70)
    print(f"NaN/vuoto responses nel run seed=42 ({nan_resp.sum()} item):")
    print("=" * 70)
    nan_items = df_new[nan_resp][["resp_s", "ic_f"]].reset_index()
    print(nan_items.to_string())
    print()

# Confronto per posizione: run seed vs run no-seed
print("=" * 70)
print("Confronto per posizione: prev vs seed=42 (solo discordanze)")
print("=" * 70)
common_pos = df_prev.index.intersection(df_new.index)
discordanti = []
for pos in sorted(common_pos):
    f1_prev = df_prev.loc[pos, "ic_f"] if pos in df_prev.index else None
    f1_new  = df_new.loc[pos, "ic_f"]  if pos in df_new.index  else None
    r_prev  = str(df_prev.loc[pos, "resp_s"])[:60] if pos in df_prev.index else "N/A"
    r_new   = str(df_new.loc[pos, "resp_s"])[:60]  if pos in df_new.index  else "N/A"
    if f1_prev != f1_new:
        discordanti.append((pos, f1_prev, f1_new, r_prev, r_new))

print(f"Posizioni con risultato diverso: {len(discordanti)}/80")
print()
print(f"{'pos':>4} {'prev':>6} {'new':>6}  risposta-prev (60ch)  |  risposta-new (60ch)")
print("-" * 100)
for pos, fp, fn, rp, rn in discordanti[:30]:
    print(f"  {pos:3d}  {fp:6.2f}  {fn:6.2f}  {rp!r:40s}  |  {rn!r}")

# Per run seed=42: categorie per item F1=0
print()
print("=" * 70)
print("Run seed=42: quali categorie hanno piu' F1=0?")
print("=" * 70)
# Dobbiamo rileggere il CSV diretto per avere la categoria
csv_new = pd.read_csv(os.path.join(RESULTS_DIR, run_new))
q30_new_full = csv_new[
    (csv_new["model"] == "qwen3-30b-a3b") & (csv_new["benchmark"] == "cuad")
].copy()
q30_new_full["ic_f"] = q30_new_full["is_correct"].astype(float)
if "category" in q30_new_full.columns:
    for cat, grp in q30_new_full.groupby("category"):
        score = grp["ic_f"].mean()
        print(f"  {cat:<25}: {score:.1%}  ({grp['ic_f'].eq(1).sum()}/{len(grp)} correct)")
else:
    print("  Colonna 'category' non disponibile nel CSV")
    print(f"  Colonne disponibili: {list(q30_new_full.columns)}")
