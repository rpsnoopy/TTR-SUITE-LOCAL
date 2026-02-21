"""
TTR-SUITE Benchmark Suite — Checkpoint manager

Persists completed task results to a JSON file so that interrupted runs
can be resumed without repeating already-evaluated tasks.

File layout: ``{checkpoint_dir}/{run_id}.json``
Content:     ``{"tasks": {"<task_id>": <result_dict>, ...}}``
"""

import json
from pathlib import Path

from src.logger import setup_logger

log = setup_logger(__name__)


class CheckpointManager:
    """
    Thread-unsafe (single-process) JSON checkpoint store.

    Parameters
    ----------
    checkpoint_dir : directory where checkpoint files are stored
    run_id         : unique identifier for this benchmark run
                     (e.g. ``"20240520_143022"``)
    """

    def __init__(self, checkpoint_dir: Path | str, run_id: str):
        self._dir    = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path   = self._dir / f"{run_id}.json"
        self._data: dict[str, dict] = {}
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_done(self, task_id: str) -> bool:
        """Return True if *task_id* has already been recorded."""
        return task_id in self._data

    def mark_done(self, task_id: str, result_dict: dict) -> None:
        """Persist *result_dict* under *task_id* and flush to disk."""
        self._data[task_id] = result_dict
        self._flush()

    def load_all(self) -> list[dict]:
        """Return all recorded result dicts (order is insertion order)."""
        return list(self._data.values())

    def count(self) -> int:
        """Return the number of completed tasks."""
        return len(self._data)

    @property
    def path(self) -> Path:
        """Path of the underlying JSON file."""
        return self._path

    # ── Internals ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = raw.get("tasks", {})
                log.info(
                    "Checkpoint loaded: %d tasks from %s",
                    len(self._data),
                    self._path,
                )
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not read checkpoint %s: %s — starting fresh", self._path, exc)
                self._data = {}
        else:
            log.debug("No checkpoint file at %s — starting fresh", self._path)

    def _flush(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps({"tasks": self._data}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except OSError as exc:
            log.error("Failed to write checkpoint %s: %s", self._path, exc)
