"""
TTR-SUITE Benchmark Suite — CUAD benchmark

Scarica il dataset CUAD da HuggingFace (theatticusproject/cuad) e valuta
l'estrazione di clausole IP su 8 categorie con F1 SQuAD-style.

Il campo ``is_correct`` in ResultRecord contiene l'F1 (0-1) per istanza;
Sheet 4 dell'XLSX ne calcola la media per categoria.
"""

from __future__ import annotations

import random
import re as _re
import string
from collections import Counter

import config
from benchmarks.base import BenchmarkBase
from src.logger import setup_logger

log = setup_logger(__name__)

# HuggingFace dataset ID per CUAD
CUAD_HF_DATASET = "theatticusproject/cuad"

# Mappa categoria → pattern da cercare nel testo della domanda
CATEGORY_MATCHERS: dict[str, list[str]] = {
    "IP-Ownership-Assignment": [
        "ip ownership", "intellectual property ownership", "ip assignment",
        "ip rights", "intellectual property", "ownership of ip",
    ],
    "Non-Compete": [
        "non-compete", "non compete", "noncompete",
        "competitive activities", "compete",
    ],
    "License-Grant": [
        "license grant", "licence grant", "license to",
    ],
    "Limitation-of-Liability": [
        "limitation of liability", "limit.*liability", "liability.*limit",
        "cap on liability",
    ],
    "Indemnification": [
        "indemnif",
    ],
    "Termination-for-Convenience": [
        "termination for convenience", "terminate.*convenience",
        "convenience termination",
    ],
    "Change-of-Control": [
        "change of control", "change-of-control",
    ],
    "Audit-Rights": [
        "audit", "inspection right", "right to audit",
    ],
}


class CUADBenchmark(BenchmarkBase):
    name = "cuad"

    def download_dataset(self) -> None:
        """Dataset scaricato lazy da HF in load_sample."""
        pass

    def load_sample(self, n: int, quick: bool = False) -> list[dict]:
        """
        Scarica CUAD da HuggingFace e restituisce n item per categoria.

        Ogni item contiene:
          context   : testo contratto (troncato a 4000 char)
          question  : domanda di estrazione
          answers   : lista di risposte ground truth
          category  : una delle 8 categorie IP
          _idx      : indice globale per il task_id
        """
        from datasets import load_dataset  # lazy import

        log.info("Caricamento CUAD da HuggingFace (%s)...", CUAD_HF_DATASET)
        load_kwargs = dict(verification_mode="no_checks", trust_remote_code=True)
        try:
            ds = load_dataset(CUAD_HF_DATASET, split="test", **load_kwargs)
        except Exception:  # noqa: BLE001
            log.info("Split 'test' non trovato, provo 'train'...")
            ds = load_dataset(CUAD_HF_DATASET, split="train", **load_kwargs)

        # Costruisce bucket per categoria
        buckets: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_MATCHERS}

        for row in ds:
            question = row.get("question", "")
            cat = _match_category(question)
            if cat is None:
                continue

            # Gestisce sia formato SQuAD (answers dict) che lista piatta
            raw_answers = row.get("answers", {})
            if isinstance(raw_answers, dict):
                answer_texts = raw_answers.get("text", [])
            elif isinstance(raw_answers, list):
                answer_texts = raw_answers
            else:
                answer_texts = []

            answers = [a for a in answer_texts if isinstance(a, str) and a.strip()]
            if not answers:
                answers = ["NESSUNA CLAUSOLA PRESENTE"]

            context = row.get("context", "")[:4000]

            buckets[cat].append({
                "context":  context,
                "question": question,
                "answers":  answers,
                "category": cat,
            })

        # Campiona n item per categoria
        items: list[dict] = []
        idx = 0
        for cat, bucket in buckets.items():
            if not bucket:
                log.warning("CUAD: nessun item per categoria '%s'", cat)
                continue
            k = min(n, len(bucket))
            sample = random.sample(bucket, k)
            for item in sample:
                item["_idx"] = idx
                idx += 1
            items.extend(sample)

        log.info("CUAD: %d item caricati in totale", len(items))
        return items

    def build_prompt(self, item: dict) -> str:
        category = item.get("category", "")
        context  = item.get("context", "")
        return (
            f"Dal seguente contratto, estrai il testo rilevante per: {category}.\n"
            "Rispondi SOLO con l'estratto esatto o "
            "'NESSUNA CLAUSOLA PRESENTE' se assente.\n\n"
            f"Contratto:\n{context}\n\n"
            "Estratto:"
        )

    def evaluate(self, prediction: str, item: dict) -> float:
        """F1 token-overlap (stile SQuAD) rispetto alla miglior risposta GT."""
        answers = item.get("answers", [])
        if not answers:
            return 0.0
        return max(compute_f1(prediction, gt) for gt in answers)


# ── F1 scoring (SQuAD-style) ───────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in s.split() if t not in ("a", "an", "the")]
    return " ".join(tokens)


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens   = _normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common   = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


# ── Category matching ─────────────────────────────────────────────────────────

def _match_category(question: str) -> str | None:
    q_lower = question.lower()
    for cat, patterns in CATEGORY_MATCHERS.items():
        for pattern in patterns:
            if _re.search(pattern, q_lower):
                return cat
    return None
