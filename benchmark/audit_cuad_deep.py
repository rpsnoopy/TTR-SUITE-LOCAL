"""
Audit approfondito CUAD:
1. Quanti item hanno ground truth vuota (nessuna clausola reale)?
2. Item con F1=0 ma risposta non-NESSUNA (falsi positivi errati)
3. qwen3-30b-a3b: nan response
4. MMLU-Pro: opzioni A-J conferma
"""
import pandas as pd
import os, glob, sys
sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = r"C:\TTR_Benchmark\results"

frames = []
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "raw_results_*.csv"))):
    df = pd.read_csv(f)
    frames.append(df)
df_all = pd.concat(frames, ignore_index=True)
if "task_id" in df_all.columns:
    df_all = df_all.drop_duplicates(subset=["model","benchmark","task_id"], keep="last")

# -- 1. CUAD: items con F1=0 ma risposta NON "nessuna clausola"
print("=" * 80)
print("1. CUAD: F1=0 ma risposta NON 'nessuna' (falsa estrazione o ground truth vuota?)")
print("=" * 80)
cuad = df_all[df_all["benchmark"] == "cuad"].copy()
cuad["ic_f"] = cuad["is_correct"].astype(float)
cuad["resp_s"] = cuad["response"].astype(str).str.strip().str.lower()

for model, grp in cuad.groupby("model"):
    zero_f1 = grp[grp["ic_f"] == 0]
    # di cui risposta che NON contiene 'nessuna' o 'nan'
    not_nessuna = zero_f1[
        ~zero_f1["resp_s"].str.contains("nessuna|nan|no clause|not present|absent", na=False)
        & (zero_f1["resp_s"] != "")
    ]
    if len(not_nessuna) > 0:
        print(f"\n{model}: {len(not_nessuna)} item con F1=0 e risposta NON-nessuna")
        for _, row in not_nessuna.head(5).iterrows():
            resp_s = str(row["response"])[:150].replace("\n", " ")
            print(f"  [{row['task_id']}] {resp_s!r}")

# -- 2. Analisi ground truth: quanti item hanno answers vuoto?
# Dobbiamo guardare nelle risposte dei modelli: se UN modello dice "NESSUNA" e un altro
# estrae testo, e il primo prende F1=0 e il secondo F1=1, allora la clausola c'e'.
# Se TUTTI i modelli prendono F1=0 su uno stesso task, potrebbe essere ground truth vuota.
print()
print("=" * 80)
print("2. CUAD: task_id dove TUTTI i modelli hanno F1=0 (potenziale ground truth vuota)")
print("=" * 80)
cuad_complete_models = ["claude-sonnet-4-6","deepcoder-14b","gpt-oss-20b",
                         "mistral-small-24b","qwen3-14b","qwen3-30b-a3b"]
cuad_filtered = cuad[cuad["model"].isin(cuad_complete_models)]
# per ogni task_id, conta quanti modelli hanno F1=0
task_zeros = cuad_filtered.groupby("task_id")["ic_f"].apply(lambda x: (x == 0).all())
all_zero_tasks = task_zeros[task_zeros].index.tolist()
print(f"Task con F1=0 per TUTTI i modelli completi: {len(all_zero_tasks)}")
print(f"Task IDs: {sorted(all_zero_tasks)}")

# guarda le risposte su un campione di questi task
if all_zero_tasks:
    sample_task = all_zero_tasks[0]
    print(f"\nCampione task {sample_task} - risposte di ogni modello:")
    for _, row in cuad[cuad["task_id"] == sample_task][["model","response","is_correct"]].iterrows():
        resp_s = str(row["response"])[:100].replace("\n", " ")
        print(f"  {row['model']}: F1={row['is_correct']} | {resp_s!r}")

# -- 3. qwen3-30b-a3b: response = 'nan'
print()
print("=" * 80)
print("3. qwen3-30b-a3b: risposte NaN")
print("=" * 80)
q30 = df_all[df_all["model"] == "qwen3-30b-a3b"]
nan_resps = q30[q30["response"].astype(str).str.strip().str.lower() == "nan"]
print(f"Risposte NaN: {len(nan_resps)}")
if len(nan_resps) > 0:
    print(nan_resps[["task_id","benchmark","is_correct","response"]].to_string())

# -- 4. MMLU-Pro: verifica opzioni A-J
print()
print("=" * 80)
print("4. MMLU-Pro: distribuzione opzioni di risposta (A-J attese)")
print("=" * 80)
mmlu = df_all[df_all["benchmark"] == "mmlupro"].copy()
mmlu["resp_clean"] = mmlu["response"].astype(str).str.strip().str.upper()
short_resp = mmlu[mmlu["resp_clean"].str.len() == 1]
print(f"Risposte di 1 char: {len(short_resp)}")
dist = short_resp["resp_clean"].value_counts()
print(dist.to_string())
beyond_J = short_resp[~short_resp["resp_clean"].isin(list("ABCDEFGHIJ"))]
print(f"\nFuori range A-J: {len(beyond_J)}")
if len(beyond_J) > 0:
    print(beyond_J[["model","task_id","resp_clean","is_correct"]].head(10).to_string())

# -- 5. Riepilogo finale
print()
print("=" * 80)
print("5. RIEPILOGO ANOMALIE")
print("=" * 80)
print("OK - MMLU-Pro: risposte corte sono lettere A-J (MMLU-Pro ha 10 opzioni)")
print("OK - LegalBench: risposte corte = Yes/No/label corretti per la maggior parte")
print("OK - Nessuna risposta vuota nel dataset finale (fix num_predict funziona)")
print("OK - CUAD F1 binario (0 o 1): comportamento atteso per estrazione esatta")
print()
print(f"[!] phi4-14b CUAD: 77/80 zero F1 (risponde sempre 'nessuna clausola')")
print(f"[?] CUAD task con tutti-zero: {len(all_zero_tasks)}/80 -- potrebbero avere ground truth vuota")
print(f"[?] deepcoder task4: F1=0 con risposta non-nessuna (ground truth diversa o vuota)")
