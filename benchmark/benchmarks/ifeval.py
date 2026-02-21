"""
TTR-SUITE Benchmark Suite — IFEval benchmark

Evaluates instruction-following using the google/IFEval dataset.
Each prompt contains a list of verifiable constraints (instruction_id_list).

Metrics
-------
- prompt_level_strict_acc : ALL constraints in the prompt satisfied
- inst_level_strict_acc   : each individual constraint satisfied

The ResultRecord.is_correct reflects prompt-level accuracy.
"""

from __future__ import annotations

import json
import re
import string
from typing import Callable

import config
from benchmarks.base import BenchmarkBase
from src.logger import setup_logger

log = setup_logger(__name__)


class IFEvalBenchmark(BenchmarkBase):
    name = "ifeval"

    def download_dataset(self) -> None:
        """Dataset is downloaded lazily via HF datasets on first load."""
        pass  # load_sample handles it

    def load_sample(self, n: int, quick: bool = False) -> list[dict]:
        from datasets import load_dataset  # lazy import

        log.info("Loading IFEval from HuggingFace (google/IFEval)…")
        ds = load_dataset(config.IFEVAL_HF_DATASET, split="train")

        items = []
        for idx, row in enumerate(ds):
            if len(items) >= n:
                break
            items.append({
                "_idx":                idx,
                "prompt":              row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs":              row.get("kwargs", []),
                "category":            "instruction-following",
            })
        return items

    def build_prompt(self, item: dict) -> str:
        return item["prompt"]

    def evaluate(self, prediction: str, item: dict) -> float:
        """
        Return 1.0 if ALL constraints are satisfied, else 0.0.
        (prompt-level strict accuracy)
        """
        instruction_ids = item.get("instruction_id_list", [])
        kwargs_list     = item.get("kwargs", [{}] * len(instruction_ids))

        if not instruction_ids:
            return 1.0

        for inst_id, kwargs in zip(instruction_ids, kwargs_list):
            if not _check_instruction(inst_id, prediction, kwargs or {}):
                return 0.0
        return 1.0

    def compute_inst_level_acc(self, prediction: str, item: dict) -> list[bool]:
        """Return per-instruction satisfaction flags (for detailed logging)."""
        instruction_ids = item.get("instruction_id_list", [])
        kwargs_list     = item.get("kwargs", [{}] * len(instruction_ids))
        return [
            _check_instruction(inst_id, prediction, kwargs or {})
            for inst_id, kwargs in zip(instruction_ids, kwargs_list)
        ]


# ── Instruction verifiers ──────────────────────────────────────────────────────

def _check_instruction(inst_id: str, response: str, kwargs: dict) -> bool:
    """Dispatch to the appropriate verifier by instruction ID prefix."""
    verifier = _VERIFIER_MAP.get(inst_id)
    if verifier is None:
        # Unknown instruction type — search by prefix
        for key, fn in _VERIFIER_MAP.items():
            if inst_id.startswith(key):
                verifier = fn
                break
    if verifier is None:
        log.debug("No verifier for instruction_id '%s' — skipping", inst_id)
        return True   # give benefit of the doubt for unknown constraints
    try:
        return verifier(response, kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("Verifier error for '%s': %s", inst_id, exc)
        return False


# ─── Individual verifiers ──────────────────────────────────────────────────────

def _verify_word_count(response: str, kwargs: dict) -> bool:
    words       = response.split()
    num_words   = len(words)
    relation    = kwargs.get("relation", "at least")
    target      = int(kwargs.get("num_words", 0))
    if relation in ("at least", "more than"):
        return num_words >= target
    if relation in ("at most", "less than", "fewer than"):
        return num_words <= target
    if relation == "exactly":
        return num_words == target
    return True


def _verify_response_language(response: str, kwargs: dict) -> bool:
    try:
        from langdetect import detect
        lang = detect(response)
        target = kwargs.get("language", "en")
        return lang == target
    except Exception:  # noqa: BLE001
        return True  # langdetect failure → skip


def _verify_json_format(response: str, kwargs: dict) -> bool:
    cleaned = response.strip()
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        json.loads(cleaned)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _verify_bullet_points(response: str, kwargs: dict) -> bool:
    bullet_pattern = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
    bullets = bullet_pattern.findall(response)
    num_bullets = len(bullets)
    relation    = kwargs.get("relation", "at least")
    target      = int(kwargs.get("num_bullets", 0))
    if relation in ("at least", "more than"):
        return num_bullets >= target
    if relation in ("at most", "less than"):
        return num_bullets <= target
    if relation == "exactly":
        return num_bullets == target
    return True


def _verify_forbidden_words(response: str, kwargs: dict) -> bool:
    forbidden: list[str] = kwargs.get("forbidden_words", [])
    resp_lower = response.lower()
    return all(word.lower() not in resp_lower for word in forbidden)


def _verify_include_keywords(response: str, kwargs: dict) -> bool:
    keywords: list[str] = kwargs.get("keywords", [])
    resp_lower = response.lower()
    return all(kw.lower() in resp_lower for kw in keywords)


def _verify_sentence_count(response: str, kwargs: dict) -> bool:
    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]
    count     = len(sentences)
    relation  = kwargs.get("relation", "at least")
    target    = int(kwargs.get("num_sentences", 0))
    if relation in ("at least", "more than"):
        return count >= target
    if relation in ("at most", "less than"):
        return count <= target
    if relation == "exactly":
        return count == target
    return True


def _verify_paragraph_count(response: str, kwargs: dict) -> bool:
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    count      = len(paragraphs)
    relation   = kwargs.get("relation", "at least")
    target     = int(kwargs.get("num_paragraphs", 0))
    if relation in ("at least", "more than"):
        return count >= target
    if relation in ("at most", "less than"):
        return count <= target
    if relation == "exactly":
        return count == target
    return True


def _verify_uppercase(response: str, kwargs: dict) -> bool:
    """Check entire response is uppercase."""
    return response == response.upper()


def _verify_lowercase(response: str, kwargs: dict) -> bool:
    """Check entire response is lowercase."""
    return response == response.lower()


def _verify_starts_with(response: str, kwargs: dict) -> bool:
    starter = kwargs.get("starter", "")
    return response.strip().startswith(starter)


def _verify_ends_with(response: str, kwargs: dict) -> bool:
    ending = kwargs.get("ending", "")
    return response.strip().endswith(ending)


def _verify_no_comma(response: str, kwargs: dict) -> bool:
    return "," not in response


def _verify_title_case(response: str, kwargs: dict) -> bool:
    words = response.split()
    return all(
        w[0].isupper() if w and w[0] not in string.punctuation else True
        for w in words
    )


def _verify_repeat_prompt(response: str, kwargs: dict) -> bool:
    """Check the response starts with the original prompt."""
    prompt = kwargs.get("original_prompt", "")
    return response.strip().startswith(prompt.strip()) if prompt else True


# ── Dispatch map ───────────────────────────────────────────────────────────────
# Keys match instruction_id strings from the IFEval dataset.

_VERIFIER_MAP: dict[str, Callable[[str, dict], bool]] = {
    # Word count
    "length_constraints:number_words":        _verify_word_count,
    "length_constraints:word_count":          _verify_word_count,
    # Language
    "language:response_language":             _verify_response_language,
    # Format
    "detectable_format:json_format":          _verify_json_format,
    "detectable_format:json":                 _verify_json_format,
    "detectable_format:number_bullet_lists":  _verify_bullet_points,
    "detectable_format:bullet_points":        _verify_bullet_points,
    "detectable_format:title_case":           _verify_title_case,
    "detectable_format:no_comma":             _verify_no_comma,
    # Keywords
    "keywords:forbidden_words":               _verify_forbidden_words,
    "keywords:existence":                     _verify_include_keywords,
    "keywords:include_keywords":              _verify_include_keywords,
    # Length
    "length_constraints:number_sentences":    _verify_sentence_count,
    "length_constraints:number_paragraphs":   _verify_paragraph_count,
    # Case
    "change_case:capital_word_frequency":     _verify_uppercase,
    "change_case:english_capital":            _verify_uppercase,
    "change_case:english_lowercase":          _verify_lowercase,
    # Start/end
    "startend:starter":                       _verify_starts_with,
    "startend:end_checker":                   _verify_ends_with,
    # Repetition
    "combination:repeat_prompt":              _verify_repeat_prompt,
}
