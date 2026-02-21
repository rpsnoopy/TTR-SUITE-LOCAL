"""
TTR-SUITE Benchmark Suite — Abstract benchmark base class

All concrete benchmarks inherit from :class:`BenchmarkBase` and implement
the five abstract methods below.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.logger import setup_logger
from src.output import ResultRecord

if TYPE_CHECKING:
    from src.checkpoint import CheckpointManager
    from src.ollama_client import OllamaClient

log = setup_logger(__name__)


class BenchmarkBase(ABC):
    """
    Abstract base for TTR-SUITE benchmarks.

    Subclasses must implement:
      - ``download_dataset``
      - ``load_sample``
      - ``build_prompt``
      - ``evaluate``

    The ``run`` method orchestrates the full evaluation loop and handles
    checkpointing, progress logging, and result collection.
    """

    #: Short name used in CSV/XLSX (e.g. ``"legalbench"``)
    name: str = ""

    def __init__(self) -> None:
        self._log = setup_logger(f"benchmark.{self.name or self.__class__.__name__}")

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def download_dataset(self) -> None:
        """
        Ensure the dataset is available locally.
        Clone from git / download from HF Hub / etc.
        Must be idempotent (skip if already present).
        """

    @abstractmethod
    def load_sample(self, n: int, quick: bool = False) -> list[dict]:
        """
        Return *n* items from the dataset.

        Each item must be a plain dict that ``build_prompt`` and
        ``evaluate`` know how to consume.
        """

    @abstractmethod
    def build_prompt(self, item: dict) -> str:
        """Format *item* as a string prompt to send to the model."""

    @abstractmethod
    def evaluate(self, prediction: str, item: dict) -> float:
        """
        Compare *prediction* with the ground truth in *item*.

        Return a float in [0, 1]:
          - Exact-match benchmarks return 0.0 or 1.0.
          - F1-based benchmarks (CUAD) return a continuous score.

        The value is stored in ``ResultRecord.is_correct`` (cast to bool
        for exact-match or kept as float for F1).
        """

    # ── Concrete run loop ──────────────────────────────────────────────────────

    def run(
        self,
        model_name: str,
        model_tag: str,
        client: "OllamaClient",
        checkpoint: "CheckpointManager",
        quick: bool = False,
        thinking: bool = False,
    ) -> list[ResultRecord]:
        """
        Evaluate this benchmark for *model_tag* and return all
        :class:`ResultRecord` instances (including already-checkpointed ones).

        Parameters
        ----------
        model_name : CLI-friendly model name stored in ResultRecord.model
        model_tag  : Ollama tag used for actual API calls
        client     : :class:`OllamaClient` instance
        checkpoint : :class:`CheckpointManager` to skip already-done tasks
        quick      : if True, use reduced sample size
        thinking   : if True, activate Qwen3 thinking mode
        """
        n_normal, n_quick = self._sample_sizes()
        n = n_quick if quick else n_normal

        items = self.load_sample(n, quick=quick)
        results: list[ResultRecord] = []

        # Re-hydrate already completed records from checkpoint
        for existing in checkpoint.load_all():
            task_id = existing.get("task_id", "")
            if existing.get("benchmark") == self.name and existing.get("model") == model_name:
                results.append(ResultRecord(**existing))

        self._log.info(
            "[%s | %s] Starting — %d items, quick=%s, thinking=%s",
            self.name, model_name, len(items), quick, thinking,
        )

        for item in items:
            task_id = self._task_id(model_name, item)

            if checkpoint.is_done(task_id):
                self._log.debug("Skip (checkpointed): %s", task_id)
                continue

            prompt = self.build_prompt(item)
            messages = [{"role": "user", "content": prompt}]

            try:
                result = client.chat(model_tag, messages, thinking=thinking)
            except RuntimeError as exc:
                self._log.error("Chat error for task %s: %s", task_id, exc)
                continue

            score = self.evaluate(result.response_text, item)
            # For exact-match, score is 0/1 → cast to bool.
            # For F1, we store the float directly in is_correct field.
            is_correct_val: bool = bool(score > 0.5)  # threshold for bool field

            record = ResultRecord(
                model=model_name,
                benchmark=self.name,
                task_id=task_id,
                category=self._item_category(item),
                prompt=prompt[:2000],        # truncate for CSV readability
                response=result.response_text[:2000],
                ground_truth=str(self._item_gt(item))[:500],
                is_correct=is_correct_val,
                time_ms=result.time_ms,
                tokens_generated=result.tokens_generated,
                tok_s=result.tok_s,
                thinking_tokens=result.thinking_tokens,
            )

            checkpoint.mark_done(task_id, dataclasses.asdict(record))
            results.append(record)

            self._log.info(
                "[%s | %s] %s → %s (%.0f ms, %.1f tok/s)",
                self.name,
                model_name,
                task_id,
                "OK" if is_correct_val else "WRONG",
                result.time_ms,
                result.tok_s,
            )

        n_correct = sum(1 for r in results if r.is_correct)
        self._log.info(
            "[%s | %s] Done — %d/%d correct (%.1f%%)",
            self.name,
            model_name,
            n_correct,
            len(results),
            100 * n_correct / len(results) if results else 0,
        )
        return results

    # ── Helpers subclasses may override ───────────────────────────────────────

    def _sample_sizes(self) -> tuple[int, int]:
        """Return (normal_n, quick_n).  Override for custom sizes."""
        from config import SAMPLE_SIZES
        return SAMPLE_SIZES.get(self.name, (20, 5))

    def _task_id(self, model_name: str, item: dict) -> str:
        """Unique task identifier for checkpointing."""
        return f"{self.name}::{model_name}::{item.get('_idx', id(item))}"

    def _item_category(self, item: dict) -> str:
        return item.get("category", "")

    def _item_gt(self, item: dict) -> str:
        return item.get("answer", item.get("ground_truth", ""))
