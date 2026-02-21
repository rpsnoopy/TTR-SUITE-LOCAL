"""
TTR-SUITE Benchmark Suite — Main entry point

Usage
-----
  python benchmark_runner.py [OPTIONS]

  --models        qwen3-32b qwen3-30b-a3b mistral-small-24b
  --benchmarks    legalbench cuad ifeval mmlupro
  --quick         Reduced sample (10-12 tasks per benchmark)
  --resume        Resume from existing checkpoint (same run_id)
  --output-dir    Override output directory
  --no-pull       Skip ollama pull (models already present)
  --dry-run       Infrastructure smoke-test (no Ollama calls)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# ── Ensure project root is on sys.path ─────────────────────────────────────────
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.anthropic_client import AnthropicClient
from src.checkpoint import CheckpointManager
from src.logger import setup_logger
from src.ollama_client import OllamaClient
from src.output import ResultRecord, ResultsWriter, generate_xlsx

if TYPE_CHECKING:
    from benchmarks.base import BenchmarkBase

log = setup_logger("benchmark_runner", config.LOGS_DIR / "benchmark_runner.log")

# ── Benchmark registry ─────────────────────────────────────────────────────────

def _get_benchmark(name: str) -> "BenchmarkBase":
    if name == "legalbench":
        from benchmarks.legalbench import LegalBenchBenchmark
        return LegalBenchBenchmark()
    if name == "cuad":
        from benchmarks.cuad import CUADBenchmark
        return CUADBenchmark()
    if name == "ifeval":
        from benchmarks.ifeval import IFEvalBenchmark
        return IFEvalBenchmark()
    if name == "mmlupro":
        from benchmarks.mmlupro import MMLUProBenchmark
        return MMLUProBenchmark()
    raise ValueError(f"Unknown benchmark: {name!r}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark_runner",
        description="TTR-SUITE LLM Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=config.DEFAULT_MODELS,
        choices=list(config.MODELS.keys()),
        metavar="MODEL",
        help=(
            "Models to benchmark. "
            f"Available: {', '.join(config.MODELS)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=config.DEFAULT_BENCHMARKS,
        choices=config.DEFAULT_BENCHMARKS,
        metavar="BENCH",
        help=(
            "Benchmarks to run. "
            f"Available: {', '.join(config.DEFAULT_BENCHMARKS)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced sample sizes (10-12 tasks per benchmark).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint (prompts for run_id).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run ID for checkpointing (default: timestamp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RESULTS_DIR,
        help=f"Output directory for CSV/XLSX. Default: {config.RESULTS_DIR}",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip 'ollama pull' (assume models are already available).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke-test infrastructure without calling Ollama.",
    )
    return parser.parse_args()


# ── Dry-run smoke test ─────────────────────────────────────────────────────────

def run_dry_run(args: argparse.Namespace) -> int:
    """Run infrastructure checks.  Returns 0 on full success, 1 otherwise."""
    checks_passed = 0
    checks_failed = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal checks_passed, checks_failed
        status = "OK" if ok else "FAIL"
        suffix = f"  [{detail}]" if detail else ""
        print(f"  [{status}] {label}{suffix}")
        if ok:
            checks_passed += 1
        else:
            checks_failed += 1

    print("\n=== TTR-SUITE Benchmark Suite — Dry Run ===\n")

    # 1. Import checks
    try:
        import benchmarks.legalbench   # noqa: F401
        import benchmarks.cuad         # noqa: F401
        import benchmarks.ifeval       # noqa: F401
        import benchmarks.mmlupro      # noqa: F401
        check("Module imports", True)
    except ImportError as exc:
        check("Module imports", False, str(exc))

    # 2. Ollama connectivity
    client = OllamaClient(config.OLLAMA_BASE_URL, config.OLLAMA_TIMEOUT_S)
    check("Ollama reachable", client.health_check(), config.OLLAMA_BASE_URL)

    # 3. Output directory creation
    try:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        check("Output directory", True, str(output_dir))
    except OSError as exc:
        check("Output directory", False, str(exc))

    # 4. Checkpoint directory
    try:
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        check("Checkpoint directory", True, str(config.CHECKPOINT_DIR))
    except OSError as exc:
        check("Checkpoint directory", False, str(exc))

    # 5. Single offline LegalBench task (no Ollama call)
    try:
        from benchmarks.legalbench import LegalBenchBenchmark, _normalize
        bench = LegalBenchBenchmark()
        fake_item = {
            "text":      "Is a cat a dog?",
            "answer":    "no",
            "task_name": "dry-run-test",
            "category":  "issue-spotting",
            "_idx":      0,
        }
        prompt = bench.build_prompt(fake_item)
        hardcoded_response = "no"
        score = bench.evaluate(hardcoded_response, fake_item)
        check("LegalBench offline task", score == 1.0, f"score={score}")
    except Exception as exc:  # noqa: BLE001
        check("LegalBench offline task", False, str(exc))

    # 6. ResultsWriter + CSV creation
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = ResultsWriter(Path(tmp), "dry_run_test.csv")
            record = ResultRecord(
                model="dry-run-model",
                benchmark="legalbench",
                task_id="dry::0",
                category="issue-spotting",
                prompt="Is a cat a dog?",
                response="no",
                ground_truth="no",
                is_correct=True,
                time_ms=0,
                tokens_generated=1,
                tok_s=0.0,
                thinking_tokens=0,
            )
            writer.write_record(record)
            writer.close()
        check("ResultsWriter CSV", True)
    except Exception as exc:  # noqa: BLE001
        check("ResultsWriter CSV", False, str(exc))

    print(f"\nResult: {checks_passed} passed, {checks_failed} failed\n")
    return 0 if checks_failed == 0 else 1


# ── Summary table ──────────────────────────────────────────────────────────────

def _print_summary(all_records: list[ResultRecord], benchmarks: list[str], models: list[str]) -> None:
    """Print an ASCII table of accuracy per model × benchmark."""
    import collections

    # Aggregate: (model, benchmark) → [is_correct, ...]
    data: dict[tuple[str, str], list[bool]] = collections.defaultdict(list)
    for r in all_records:
        data[(r.model, r.benchmark)].append(r.is_correct)

    col_w = 12
    model_w = 28

    # Header
    header = f"{'Model':<{model_w}}" + "".join(f"{b.upper()[:col_w]:>{col_w}}" for b in benchmarks)
    sep    = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(" TTR-SUITE Benchmark Results")
    print(sep)
    print(header)
    print(sep)

    for model_name in models:
        row = f"{model_name:<{model_w}}"
        for bench in benchmarks:
            vals = data.get((model_name, bench), [])
            if vals:
                acc = 100 * sum(vals) / len(vals)
                cell = f"{acc:.1f}%"
            else:
                cell = "—"
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep + "\n")


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # Ensure logs directory exists
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return run_dry_run(args)

    # Determine run_id
    if args.run_id:
        run_id = args.run_id
    elif args.resume:
        # List existing checkpoints and let the user pick
        checkpoint_files = sorted(config.CHECKPOINT_DIR.glob("*.json"))
        if not checkpoint_files:
            log.error("No checkpoint files found in %s", config.CHECKPOINT_DIR)
            return 1
        print("Available checkpoints:")
        for i, f in enumerate(checkpoint_files):
            print(f"  [{i}] {f.stem}")
        idx_str = input("Select checkpoint index: ").strip()
        run_id = checkpoint_files[int(idx_str)].stem
        log.info("Resuming run_id: %s", run_id)
    else:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output setup
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    csv_filename = f"raw_results_{run_id}.csv"
    checkpoint   = CheckpointManager(config.CHECKPOINT_DIR, run_id)
    # Client Ollama condiviso per tutti i modelli locali
    ollama_client = OllamaClient(config.OLLAMA_BASE_URL, config.OLLAMA_TIMEOUT_S)
    all_records: list[ResultRecord] = []

    # Re-hydrate from checkpoint on resume
    if args.resume:
        for rec_dict in checkpoint.load_all():
            try:
                all_records.append(ResultRecord(**rec_dict))
            except TypeError:
                pass
        log.info("Loaded %d records from checkpoint", len(all_records))

    with ResultsWriter(output_dir, csv_filename) as writer:
        # Write any resumed records back to CSV (if file is new)
        for rec in all_records:
            writer.write_record(rec)

        for model_name in args.models:
            model_cfg = config.MODELS[model_name]
            provider  = model_cfg.get("provider", "ollama")
            thinking  = model_cfg.get("thinking", False)

            # ── Seleziona il client corretto ───────────────────────────────
            if provider == "anthropic":
                model_id = model_cfg["model_id"]
                client   = AnthropicClient(model_id, config.OLLAMA_TIMEOUT_S)
                model_tag = model_id          # usato nei log e nel task_id
                log.info("=== Model: %s (%s) [Anthropic API] ===", model_name, model_id)

                if not client.health_check():
                    log.error(
                        "Anthropic health check fallito per %s — skipping. "
                        "Imposta ANTHROPIC_API_KEY.", model_name
                    )
                    continue

            else:  # ollama
                model_tag = model_cfg["tag"]
                client    = ollama_client
                log.info("=== Model: %s (%s) [Ollama] ===", model_name, model_tag)

                # Pull model unless --no-pull
                if not args.no_pull:
                    try:
                        client.pull_model(model_tag)
                    except RuntimeError as exc:
                        log.error("Pull failed: %s — skipping model", exc)
                        continue

            for bench_name in args.benchmarks:
                log.info("--- Benchmark: %s ---", bench_name)
                bench = _get_benchmark(bench_name)

                try:
                    bench.download_dataset()
                except Exception as exc:  # noqa: BLE001
                    log.error("Dataset download failed for %s: %s", bench_name, exc)
                    continue

                try:
                    records = bench.run(
                        model_name=model_name,
                        model_tag=model_tag,
                        client=client,
                        checkpoint=checkpoint,
                        quick=args.quick,
                        thinking=thinking,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error("Benchmark %s failed for %s: %s", bench_name, model_name, exc)
                    continue

                for record in records:
                    if not args.resume or record not in all_records:
                        writer.write_record(record)
                    all_records.append(record)

            # Libera VRAM solo per modelli Ollama
            if provider == "ollama":
                ollama_client.stop_model(model_tag)

    # Generate XLSX summary
    csv_path = output_dir / csv_filename
    if csv_path.exists():
        try:
            xlsx_path = generate_xlsx(csv_path)
            log.info("Summary XLSX: %s", xlsx_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not generate XLSX: %s", exc)

    # Print ASCII summary table
    _print_summary(all_records, args.benchmarks, args.models)

    log.info("Run %s complete.  Results in: %s", run_id, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
