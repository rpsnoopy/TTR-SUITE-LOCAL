"""
TTR-SUITE Benchmark Suite — Results writer

Writes raw results to an append-mode CSV and generates a summary XLSX
with four sheets:

  Sheet 1 — Accuracy per model × benchmark
  Sheet 2 — Throughput (tok/s) per model, with/without thinking
  Sheet 3 — LegalBench accuracy per category (6 categories)
  Sheet 4 — CUAD F1-score per category (8 IP categories)
"""

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.logger import setup_logger

log = setup_logger(__name__)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ResultRecord:
    model:            str
    benchmark:        str
    task_id:          str
    category:         str
    prompt:           str
    response:         str
    ground_truth:     str
    is_correct:       bool
    time_ms:          int
    tokens_generated: int
    tok_s:            float
    thinking_tokens:  int    # 0 when thinking mode is off


_CSV_FIELDS = [f.name for f in dataclasses.fields(ResultRecord)]


# ── Writer ─────────────────────────────────────────────────────────────────────

class ResultsWriter:
    """
    Append-mode CSV writer.

    Opens the file immediately; creates the header row if the file is new.
    Call :meth:`write_record` for each result.  The file is flushed after
    every write so partial runs are recoverable.
    """

    def __init__(self, output_dir: Path | str, csv_filename: str):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._output_dir / csv_filename

        is_new = not self._csv_path.exists()
        self._fh = self._csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=_CSV_FIELDS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()
        log.info("ResultsWriter ready: %s", self._csv_path)

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def write_record(self, record: ResultRecord) -> None:
        self._writer.writerow(dataclasses.asdict(record))
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── XLSX generator ─────────────────────────────────────────────────────────────

def generate_xlsx(csv_path: Path | str) -> Path:
    """
    Read *csv_path* and produce a summary XLSX alongside it.

    Returns the path to the generated XLSX file.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    xlsx_path = csv_path.with_name(
        csv_path.name.replace("raw_results_", "summary_").replace(".csv", ".xlsx")
    )

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        _sheet_accuracy(df, writer)
        _sheet_throughput(df, writer)
        _sheet_legalbench_categories(df, writer)
        _sheet_cuad_categories(df, writer)

    log.info("XLSX summary written: %s", xlsx_path)
    return xlsx_path


# ── Sheet builders ─────────────────────────────────────────────────────────────

def _sheet_accuracy(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    """Sheet 1: accuracy per model × benchmark."""
    if df.empty:
        pd.DataFrame().to_excel(writer, sheet_name="Accuracy", index=False)
        return

    pivot = (
        df.groupby(["model", "benchmark"])["is_correct"]
        .mean()
        .mul(100)
        .round(2)
        .unstack(level="benchmark")
    )
    pivot.columns.name = None
    pivot.index.name   = "Model"
    pivot.to_excel(writer, sheet_name="Accuracy")


def _sheet_throughput(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    """Sheet 2: average tok/s per model (all tasks; thinking tasks separately)."""
    if df.empty:
        pd.DataFrame().to_excel(writer, sheet_name="Throughput", index=False)
        return

    overall = (
        df.groupby("model")["tok_s"]
        .mean()
        .round(2)
        .rename("avg_tok_s_all")
    )
    thinking = (
        df[df["thinking_tokens"] > 0]
        .groupby("model")["tok_s"]
        .mean()
        .round(2)
        .rename("avg_tok_s_thinking")
    )
    non_thinking = (
        df[df["thinking_tokens"] == 0]
        .groupby("model")["tok_s"]
        .mean()
        .round(2)
        .rename("avg_tok_s_no_thinking")
    )

    out = pd.concat([overall, thinking, non_thinking], axis=1)
    out.index.name = "Model"
    out.to_excel(writer, sheet_name="Throughput")


def _sheet_legalbench_categories(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    """Sheet 3: LegalBench accuracy per model × category."""
    lb = df[df["benchmark"] == "legalbench"]
    if lb.empty:
        pd.DataFrame().to_excel(writer, sheet_name="LegalBench", index=False)
        return

    pivot = (
        lb.groupby(["model", "category"])["is_correct"]
        .mean()
        .mul(100)
        .round(2)
        .unstack(level="category")
    )
    pivot.columns.name = None
    pivot.index.name   = "Model"
    pivot.to_excel(writer, sheet_name="LegalBench")


def _sheet_cuad_categories(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    """Sheet 4: CUAD F1-score per model × category.

    The ``is_correct`` column stores the per-instance F1 (0–1) for CUAD;
    we average it per category to get macro-F1.
    """
    cuad = df[df["benchmark"] == "cuad"]
    if cuad.empty:
        pd.DataFrame().to_excel(writer, sheet_name="CUAD", index=False)
        return

    pivot = (
        cuad.groupby(["model", "category"])["is_correct"]
        .mean()
        .mul(100)
        .round(2)
        .unstack(level="category")
    )
    pivot.columns.name = None
    pivot.index.name   = "Model"
    pivot.to_excel(writer, sheet_name="CUAD")
