"""
TTR-SUITE — Consolidated Results Builder
=========================================
Merges all per-run CSVs into a single comparison XLSX.

Usage:
    python consolidate_results.py [--output PATH]

Re-run whenever a new model is added (e.g. gpt-4o, llama-3.3-70b, etc.).
Models with multiple runs are averaged automatically.
Incomplete models (missing ≥1 benchmark) are listed in a "Partial" sheet.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path("C:/TTR_Benchmark/results")
BENCHMARK_RESULTS = Path(__file__).parent / "results"
BENCHMARKS   = ["legalbench", "cuad", "ifeval", "mmlupro"]

# Display order for models in the final table (add new models here)
MODEL_ORDER = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-oss",          # placeholder for future
    "mistral-small-24b",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-32b",
]


# ── Load & merge ────────────────────────────────────────────────────────────────

def load_all_csvs(results_dir: Path) -> pd.DataFrame:
    """Load every raw_results_*.csv and concatenate."""
    csv_files = sorted(results_dir.glob("raw_results_*.csv"))
    if not csv_files:
        sys.exit(f"No raw_results_*.csv found in {results_dir}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        df["_source_file"] = f.name
        frames.append(df)
        print(f"  Loaded {f.name}: {len(df)} rows, "
              f"models={sorted(df['model'].unique())}")

    return pd.concat(frames, ignore_index=True)


def classify_models(df: pd.DataFrame):
    """Split models into complete (all 4 benchmarks) and partial."""
    complete, partial = [], []
    for model, g in df.groupby("model"):
        done = set(g["benchmark"].unique())
        if set(BENCHMARKS).issubset(done):
            complete.append(model)
        else:
            partial.append((model, sorted(done)))
    return complete, partial


# ── Sheet builders ──────────────────────────────────────────────────────────────

def _sort_models(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows according to MODEL_ORDER; unknown models go last alphabetically."""
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    idx = df.index.to_series().map(lambda m: order.get(m, 999 + ord(m[0])))
    return df.loc[idx.sort_values().index]


def sheet_accuracy(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    pivot = (
        df.groupby(["model", "benchmark"])["is_correct"]
        .mean().mul(100).round(1)
        .unstack("benchmark")
        .reindex(columns=BENCHMARKS)
    )
    pivot.columns.name = None
    pivot.index.name   = "Model"
    pivot = _sort_models(pivot)
    pivot["AVG"] = pivot[BENCHMARKS].mean(axis=1).round(1)
    pivot.to_excel(writer, sheet_name="Accuracy")


def sheet_throughput(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    overall = df.groupby("model")["tok_s"].mean().round(1).rename("avg_tok_s")
    thinking = (
        df[df["thinking_tokens"] > 0].groupby("model")["tok_s"]
        .mean().round(1).rename("avg_tok_s_thinking")
    )
    non_thinking = (
        df[df["thinking_tokens"] == 0].groupby("model")["tok_s"]
        .mean().round(1).rename("avg_tok_s_no_thinking")
    )
    out = pd.concat([overall, thinking, non_thinking], axis=1)
    out.index.name = "Model"
    out = _sort_models(out)
    out.to_excel(writer, sheet_name="Throughput")


def sheet_legalbench(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    lb = df[df["benchmark"] == "legalbench"]
    if lb.empty:
        return
    pivot = (
        lb.groupby(["model", "category"])["is_correct"]
        .mean().mul(100).round(1)
        .unstack("category")
    )
    pivot.columns.name = None
    pivot.index.name   = "Model"
    pivot = _sort_models(pivot)
    pivot["AVG"] = pivot.mean(axis=1).round(1)
    pivot.to_excel(writer, sheet_name="LegalBench")


def sheet_cuad(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    cuad = df[df["benchmark"] == "cuad"]
    if cuad.empty:
        return
    pivot = (
        cuad.groupby(["model", "category"])["is_correct"]
        .mean().mul(100).round(1)
        .unstack("category")
    )
    pivot.columns.name = None
    pivot.index.name   = "Model"
    pivot = _sort_models(pivot)
    pivot["AVG"] = pivot.mean(axis=1).round(1)
    pivot.to_excel(writer, sheet_name="CUAD")


def sheet_partial(partial: list, writer: pd.ExcelWriter) -> None:
    if not partial:
        return
    rows = [{"Model": m, "Benchmarks completed": ", ".join(b)} for m, b in partial]
    pd.DataFrame(rows).to_excel(writer, sheet_name="Partial runs", index=False)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build consolidated benchmark XLSX")
    parser.add_argument("--output", type=Path,
                        default=BENCHMARK_RESULTS / f"consolidated_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                        help="Output XLSX path")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Directory containing raw_results_*.csv files")
    args = parser.parse_args()

    print(f"\nLoading CSVs from {args.results_dir} ...")
    df = load_all_csvs(args.results_dir)

    print(f"\nTotal rows: {len(df)}")
    complete, partial = classify_models(df)
    print(f"Complete models ({len(complete)}): {complete}")
    if partial:
        print(f"Partial models ({len(partial)}): {[m for m,_ in partial]}")

    # Work only on complete models for main sheets
    df_complete = df[df["model"].isin(complete)]

    # Models with multiple runs → average naturally via groupby (each row is a task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {args.output} ...")
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        sheet_accuracy(df_complete, writer)
        sheet_throughput(df_complete, writer)
        sheet_legalbench(df_complete, writer)
        sheet_cuad(df_complete, writer)
        sheet_partial(partial, writer)

    print(f"\nDone: {args.output}")

    # Print ASCII summary
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'LegalBench':>11} {'CUAD':>7} {'IFEval':>8} {'MMLU-Pro':>10} {'AVG':>6}")
    print("-" * 70)
    for model in complete:
        row = {}
        for bm in BENCHMARKS:
            sub = df_complete[(df_complete["model"] == model) & (df_complete["benchmark"] == bm)]
            row[bm] = sub["is_correct"].mean() * 100 if not sub.empty else float("nan")
        avg = sum(row.values()) / len(row)
        print(f"{model:<25} {row['legalbench']:>10.1f}% {row['cuad']:>6.1f}% "
              f"{row['ifeval']:>7.1f}% {row['mmlupro']:>9.1f}% {avg:>5.1f}%")
    if partial:
        print("-" * 70)
        for m, b in partial:
            print(f"{m:<25} (partial: {', '.join(b)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
