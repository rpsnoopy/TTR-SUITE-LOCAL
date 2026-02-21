"""
TTR-SUITE Benchmark Suite — LegalBench benchmark

Downloads the LegalBench repository and evaluates models on tasks
from 6 legal reasoning categories.

Categories
----------
- issue-spotting
- rule-recall
- rule-conclusion
- rule-application
- interpretation
- rhetorical-understanding

Each category maps to one or more task sub-directories inside the repo,
each containing a ``test.tsv`` or ``test.csv`` file with at minimum the
columns ``text`` and ``answer``.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import git  # gitpython

import config
from benchmarks.base import BenchmarkBase
from src.logger import setup_logger

log = setup_logger(__name__)

# Mapping from our 6 categories → known LegalBench task directory names.
# The actual repo contains 162 tasks; we select representative ones.
CATEGORY_TASKS: dict[str, list[str]] = {
    "issue-spotting": [
        "abercrombie",
        "learned_hands_benefits",
        "learned_hands_business",
        "learned_hands_consumer",
        "learned_hands_courts",
        "learned_hands_crime",
        "learned_hands_divorce",
        "learned_hands_domestic_violence",
        "learned_hands_education",
        "learned_hands_employment",
    ],
    "rule-recall": [
        "definition_classification",
        "hearsay",
        "insurance_policy_interpretation",
        "contract_qa",
        "rule_qa",
    ],
    "rule-conclusion": [
        "personal_jurisdiction",
        "canada_tax_court_outcomes",
        "proa",
        "scalr",
    ],
    "rule-application": [
        "citation_prediction_classification",
        "diversity_1",
        "nys_judicial_ethics",
        "corporate_lobbying",
    ],
    "interpretation": [
        "ucc_v_common_law",
        "successor_liability",
        "textualism_tool_dictionaries",
        "textualism_tool_plain",
    ],
    "rhetorical-understanding": [
        "oral_argument_question_purpose",
        "function_of_decision_section",
    ],
}


class LegalBenchBenchmark(BenchmarkBase):
    name = "legalbench"

    def __init__(self) -> None:
        super().__init__()
        self._repo_dir = config.LEGALBENCH_DIR

    # ── BenchmarkBase interface ────────────────────────────────────────────────

    def download_dataset(self) -> None:
        if self._repo_dir.exists() and (self._repo_dir / ".git").exists():
            log.info("LegalBench already cloned at %s", self._repo_dir)
            return
        log.info("Cloning LegalBench from %s …", config.LEGALBENCH_REPO_URL)
        self._repo_dir.parent.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(config.LEGALBENCH_REPO_URL, self._repo_dir, depth=1)
        log.info("LegalBench clone complete.")

    def load_sample(self, n: int, quick: bool = False) -> list[dict]:
        """
        Return *n* items total, spread across 6 categories.

        Normal:  4 items per category = 24 total
        Quick:   2 items per category = 12 total
        """
        n_per_cat = 2 if quick else 4
        items: list[dict] = []
        idx = 0

        for category, task_names in CATEGORY_TASKS.items():
            cat_items = self._load_category(category, task_names, n_per_cat)
            for item in cat_items:
                item["_idx"]     = idx
                item["category"] = category
                idx += 1
            items.extend(cat_items)

        return items

    def build_prompt(self, item: dict) -> str:
        task_name = item.get("task_name", "legal task")
        return (
            f"Task: {task_name}\n\n"
            f"{item['text']}\n\n"
            "Answer:"
        )

    def evaluate(self, prediction: str, item: dict) -> float:
        gt   = _normalize(item.get("answer", ""))
        pred = _normalize(prediction)

        # 1. Exact match
        if pred == gt:
            return 1.0
        # 2. GT is contained as a whole word in the prediction
        #    (model answered correctly but with extra explanation)
        import re
        if re.search(r'\b' + re.escape(gt) + r'\b', pred):
            return 1.0
        # 3. Prediction starts with GT (e.g. "yes, because...")
        if pred.startswith(gt):
            return 1.0
        return 0.0

    # ── Internals ──────────────────────────────────────────────────────────────

    def _load_category(
        self,
        category: str,
        task_names: list[str],
        n: int,
    ) -> list[dict]:
        """
        Try each task in *task_names* until we collect *n* items.

        The LegalBench repo layout:
          legalbench/tasks/<task_name>/base_task.csv   (or test.csv / data.csv)
        Each file has at least ``text`` and ``answer`` columns.
        """
        collected: list[dict] = []
        tasks_dir = self._repo_dir / "tasks"

        for task_name in task_names:
            if len(collected) >= n:
                break

            task_dir = tasks_dir / task_name
            if not task_dir.is_dir():
                log.debug("Task dir not found: %s", task_dir)
                continue

            data_file = _find_data_file(task_dir)
            if data_file is None:
                log.debug("No data file in: %s", task_dir)
                continue

            rows = _read_tsv_or_csv(data_file)
            need = n - len(collected)
            sample = random.sample(rows, min(need, len(rows))) if rows else []
            for row in sample:
                row["task_name"] = task_name
            collected.extend(sample)

        if len(collected) < n:
            log.warning(
                "LegalBench category '%s': only %d/%d items found",
                category, len(collected), n,
            )
        return collected[:n]


# ── File helpers ───────────────────────────────────────────────────────────────

def _find_data_file(task_dir: Path) -> Path | None:
    """Return the first CSV/TSV data file found in *task_dir*."""
    for name in ("base_task.csv", "test.csv", "data.csv", "train.csv",
                 "base_task.tsv", "test.tsv", "train.tsv"):
        candidate = task_dir / name
        if candidate.exists():
            return candidate
    # Fall back to any CSV/TSV
    for ext in ("*.csv", "*.tsv"):
        found = list(task_dir.glob(ext))
        if found:
            return found[0]
    return None


def _read_tsv_or_csv(path: Path) -> list[dict]:
    """Read a CSV or TSV into a list of dicts.  Returns [] on error."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        rows = []
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row in reader:
                # Normalise common column names
                item: dict = {}
                for key, val in row.items():
                    k = key.strip().lower()
                    item[k] = val.strip() if isinstance(val, str) else val
                # Normalise: accept 'question' as alias for 'text'
                if "question" in item and "text" not in item:
                    item["text"] = item["question"]
                # Must have 'text' and 'answer'
                if "text" in item and "answer" in item:
                    rows.append(item)
        return rows
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s: %s", path, exc)
        return []


def _normalize(s: str) -> str:
    return s.strip().lower()
