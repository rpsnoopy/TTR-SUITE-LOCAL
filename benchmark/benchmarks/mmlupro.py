"""
TTR-SUITE Benchmark Suite — MMLU-Pro benchmark (Law/Jurisprudence)

Evaluates models on multiple-choice law questions from the
TIGER-Lab/MMLU-Pro dataset.  MMLU-Pro uses 10 options (A–J) instead
of the classic 4.
"""

from __future__ import annotations

import random
import re

import config

# Fixed seed so every run samples the same law questions.
SAMPLE_SEED = 42
from benchmarks.base import BenchmarkBase
from src.logger import setup_logger

log = setup_logger(__name__)

# Option labels for MMLU-Pro (up to 10 options A–J)
OPTION_LABELS = list("ABCDEFGHIJ")


class MMLUProBenchmark(BenchmarkBase):
    name = "mmlupro"

    def download_dataset(self) -> None:
        """Dataset is downloaded lazily via HF datasets on first load."""
        pass

    def load_sample(self, n: int, quick: bool = False) -> list[dict]:
        from datasets import load_dataset  # lazy import

        log.info("Loading MMLU-Pro from HuggingFace (TIGER-Lab/MMLU-Pro)…")
        # MMLU-Pro has a 'test' split with ~12k questions
        ds = load_dataset(config.MMLUPRO_HF_DATASET, split="test")

        # Filter to law / jurisprudence categories
        law_items = []
        for idx, row in enumerate(ds):
            subject = (row.get("category") or row.get("subject") or "").lower()
            if subject in config.MMLUPRO_LAW_SUBJECTS:
                law_items.append((idx, row))

        if not law_items:
            log.warning(
                "MMLU-Pro: no items found for subjects %s — using all categories",
                config.MMLUPRO_LAW_SUBJECTS,
            )
            law_items = list(enumerate(ds))[:n]

        # Sample — seeded for reproducibility across model runs
        k = min(n, len(law_items))
        rng = random.Random(SAMPLE_SEED)
        sample = rng.sample(law_items, k)

        items = []
        for position, (orig_idx, row) in enumerate(sample):
            options: list[str] = row.get("options", [])
            answer_index: int  = row.get("answer_index", 0)

            if answer_index < len(OPTION_LABELS):
                answer_letter = OPTION_LABELS[answer_index]
            else:
                answer_letter = OPTION_LABELS[0]

            items.append({
                "_idx":         position,
                "question":     row.get("question", ""),
                "options":      options,
                "answer_index": answer_index,
                "answer":       answer_letter,
                "category":     (row.get("category") or row.get("subject") or "law"),
            })

        return items

    def build_prompt(self, item: dict) -> str:
        options: list[str] = item.get("options", [])
        lines = [f"Domanda: {item['question']}", "", "Opzioni:"]
        for i, opt in enumerate(options):
            label = OPTION_LABELS[i] if i < len(OPTION_LABELS) else str(i)
            lines.append(f"{label}) {opt}")
        lines.append("")
        lines.append(
            "Rispondi con la sola lettera dell'opzione corretta (A, B, C, …)."
        )
        return "\n".join(lines)

    def evaluate(self, prediction: str, item: dict) -> float:
        """
        Extract the predicted letter from the model response and compare
        to the ground truth.
        """
        gt = item.get("answer", "A").strip().upper()
        pred = _extract_option_letter(prediction)
        return 1.0 if pred == gt else 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_option_letter(text: str) -> str:
    """
    Extract the answer letter from a model response.

    Handles patterns like:
      - "A"  /  "A."  /  "A)"
      - "The answer is A"
      - "A) some explanation…"
    """
    text = text.strip()

    # Direct single-letter response
    if len(text) == 1 and text.upper() in OPTION_LABELS:
        return text.upper()

    # "The answer is X" pattern
    m = re.search(r"answer\s*(?:is|:)?\s*([A-Ja-j])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Letter followed by ) or .  at start of string
    m = re.match(r"^([A-Ja-j])[).]", text)
    if m:
        return m.group(1).upper()

    # First standalone uppercase letter in the response
    m = re.search(r"\b([A-J])\b", text)
    if m:
        return m.group(1).upper()

    return ""
