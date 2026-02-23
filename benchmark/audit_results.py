"""
Audit completo dei raw results: cerca anomalie per ogni modello/benchmark.
"""
import pandas as pd
import os, glob, sys

RESULTS_DIR = r"C:\TTR_Benchmark\results"

# forza UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# -- carica tutti i CSV
frames = []
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "raw_results_*.csv"))):
    df = pd.read_csv(f)
    df["_src"] = os.path.basename(f)
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)

if "task_id" in df_all.columns:
    df_all = df_all.drop_duplicates(subset=["model","benchmark","task_id"], keep="last")

print(f"Totale righe (post-dedup): {len(df_all)}")
print(f"Modelli: {sorted(df_all['model'].unique())}")
print()

# -- tabella riassuntiva per modello x benchmark
print(f"{'Modello':<25} | {'Bench':<12} | {'n':>3} | {'Score':>7} | Flags")
print("-" * 80)
for (model, bench), grp in df_all.groupby(["model","benchmark"]):
    n = len(grp)
    resp = grp["response"].astype(str)
    empty = (resp.str.strip() == "").sum()
    very_short = ((resp.str.strip().str.len() < 5) & (resp.str.strip() != "")).sum()

    if bench == "cuad":
        ic = grp["is_correct"].astype(float)
        score = ic.mean()
        zero_f1 = (ic == 0).sum()
    else:
        ic_bool = grp["is_correct"].astype(str).str.lower().isin(["true","1","1.0"])
        score = ic_bool.mean()
        zero_f1 = None

    flags = []
    if empty > n * 0.05:
        flags.append(f"EMPTY={empty}/{n}")
    if bench == "cuad" and zero_f1 is not None and zero_f1 > n * 0.80:
        flags.append(f"CUAD_ZERO_F1={zero_f1}/{n}")
    if bench in ("legalbench",) and very_short > n * 0.40:
        flags.append(f"MANY_SHORT={very_short}/{n}")

    flag_str = "  [!] " + ", ".join(flags) if flags else ""
    print(f"{model:<25} | {bench:<12} | {n:>3} | {score:>6.1%} |{flag_str}")

print()
print("=" * 80)
print("ANALISI PROFONDA: MMLUPRO - conferma che risposte corte = lettere (A/B/C/D/E)")
print("=" * 80)
for model, grp in df_all[df_all["benchmark"] == "mmlupro"].groupby("model"):
    resp = grp["response"].astype(str).str.strip()
    short = resp[resp.str.len() < 5]
    letter_answers = short[short.str.upper().isin(["A","B","C","D","E","(A)","(B)","(C)","(D)","(E)"])]
    non_letter_short = short[~short.str.upper().isin(["A","B","C","D","E","(A)","(B)","(C)","(D)","(E)"])]
    empty = (resp == "").sum()
    print(f"\n{model} | n={len(grp)}")
    print(f"  Risposte corte (<5 char): {len(short)}")
    print(f"  Di cui lettere A-E:       {len(letter_answers)}")
    print(f"  Di cui altro corto:        {len(non_letter_short)}")
    if len(non_letter_short) > 0:
        print(f"  Esempi 'altro corto': {list(non_letter_short.head(5))}")
    print(f"  Risposte vuote:            {empty}")

print()
print("=" * 80)
print("ANALISI PROFONDA: CUAD - distribuzione F1 per ogni modello")
print("=" * 80)
for model, grp in df_all[df_all["benchmark"] == "cuad"].groupby("model"):
    ic = grp["is_correct"].astype(float)
    n = len(ic)
    zero = (ic == 0).sum()
    perfect = (ic >= 0.99).sum()
    partial = ((ic > 0) & (ic < 0.99)).sum()
    print(f"\n{model} | n={n} | avg_F1={ic.mean():.3f}")
    print(f"  F1=0 (miss total):       {zero:3d} ({zero/n:.0%})")
    print(f"  F1=1 (perfect extract):  {perfect:3d} ({perfect/n:.0%})")
    print(f"  0<F1<1 (partial):        {partial:3d} ({partial/n:.0%})")
    # campione di risposte F1=0 (prime 2)
    zeros_sample = grp[ic == 0][["task_id","response"]].head(2)
    for _, row in zeros_sample.iterrows():
        resp_s = str(row["response"])[:100].replace("\n"," ")
        print(f"  MISS ex: [{row['task_id']}] {resp_s!r}")

print()
print("=" * 80)
print("ANALISI PROFONDA: LEGALBENCH - risposte corte sono corrette?")
print("=" * 80)
for model, grp in df_all[df_all["benchmark"] == "legalbench"].groupby("model"):
    resp = grp["response"].astype(str).str.strip()
    ic = grp["is_correct"].astype(str).str.lower().isin(["true","1","1.0"])
    short = resp.str.len() < 5
    print(f"\n{model} | n={len(grp)} | score={ic.mean():.1%}")
    print(f"  Risposte corte (<5):     {short.sum()}")
    print(f"  Corte CORRETTE:          {(short & ic).sum()}")
    print(f"  Corte SBAGLIATE:         {(short & ~ic).sum()}")
    # campione risposte sbagliate corte
    wrong_short = grp[short & ~ic][["task_id","response"]].head(3)
    for _, row in wrong_short.iterrows():
        print(f"  WRONG SHORT: [{row['task_id']}] {str(row['response'])!r}")

print()
print("=" * 80)
print("ANALISI PROFONDA: RISPOSTE VUOTE (EMPTY)")
print("=" * 80)
for (model, bench), grp in df_all.groupby(["model","benchmark"]):
    resp = grp["response"].astype(str)
    empty_mask = resp.str.strip() == ""
    empty_n = empty_mask.sum()
    n = len(grp)
    if empty_n > 0:
        tok_col = [c for c in grp.columns if "token" in c.lower() or "tok" in c.lower()]
        sample = grp[empty_mask].head(3)
        print(f"\n{model} | {bench} | EMPTY={empty_n}/{n}")
        for _, row in sample.iterrows():
            tok_info = " | ".join(f"{c}={row.get(c,'?')}" for c in tok_col)
            print(f"  [{row['task_id']}] {tok_info}")
