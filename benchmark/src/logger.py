"""
TTR-SUITE Benchmark Suite — Logging utility
Provides a dual-sink logger: rotating file + coloured console.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    name: str,
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Return a logger that writes to *log_file* (rotating) and to stdout.

    Parameters
    ----------
    name        : logger name (typically __name__ of the calling module)
    log_file    : path to the .log file; if None, file logging is skipped
    level       : logging threshold for both handlers
    max_bytes   : max size per log file before rotation
    backup_count: number of rotated files to keep
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return as-is to avoid duplicate handlers
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── File handler ───────────────────────────────────────────────────────────
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
