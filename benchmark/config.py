"""
TTR-SUITE Benchmark Suite — Central Configuration
All path and parameter constants live here. No hardcoded values in other modules.
"""

from pathlib import Path

# ── Directories ────────────────────────────────────────────────────────────────
BENCHMARK_ROOT = Path(__file__).parent.resolve()
DATASETS_DIR   = BENCHMARK_ROOT / "datasets"
LOGS_DIR       = BENCHMARK_ROOT / "logs"
RESULTS_DIR    = Path("C:/TTR_Benchmark/results")   # Windows-native output path

# Ensure local benchmark results/ can be a soft alias (created at runtime)
LOCAL_RESULTS_ALIAS = BENCHMARK_ROOT / "results"

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_TIMEOUT_S  = 300          # seconds — 300s per cold start + CUAD long docs
OLLAMA_KEEP_ALIVE = "0"          # release VRAM immediately after each model
OLLAMA_NUM_CTX    = 4096         # context window — max needed ~2500 tok (CUAD);
                                  # default 65536 wastes VRAM on RTX 4090 16GB

# ── Models ─────────────────────────────────────────────────────────────────────
# Maps CLI-friendly name → config dict
#
# Campi comuni:
#   provider   : "ollama" | "anthropic"
#   tag        : Ollama tag (solo per provider=ollama)
#   model_id   : Anthropic model ID (solo per provider=anthropic)
#   thinking   : attiva Qwen3 thinking mode (solo Ollama, default False)

MODELS: dict[str, dict] = {
    # ── Modelli locali (Ollama) ─────────────────────────────────────────────
    "qwen3-14b": {
        "provider": "ollama",
        "tag":      "qwen3:14b",        # 9.3 GB — entra in 16GB VRAM, pure GPU
        "thinking": False,
    },
    "qwen3-30b-a3b": {
        "provider": "ollama",
        "tag":      "qwen3:30b",        # 18 GB — split CPU/GPU su 16GB VRAM (~22 tok/s)
        "thinking": False,
    },
    "qwen3-32b": {
        "provider": "ollama",
        "tag":      "qwen3:32b-q4_K_M", # 20 GB — split CPU/GPU su 16GB VRAM (~3-22 tok/s)
        "thinking": False,
    },
    "mistral-small-24b": {
        "provider": "ollama",
        "tag":      "mistral-small:24b",
        "thinking": False,
    },
    # ── Modelli cloud (Anthropic API) ───────────────────────────────────────
    "claude-sonnet-4-5": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20251022",
        "thinking": False,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "thinking": False,
    },
}

# Modelli locali di default (esclude Claude per non richiedere API key di default)
DEFAULT_MODELS = ["qwen3-14b", "qwen3-30b-a3b", "qwen3-32b", "mistral-small-24b"]

# ── Benchmarks ─────────────────────────────────────────────────────────────────
DEFAULT_BENCHMARKS = ["legalbench", "cuad", "ifeval", "mmlupro"]

# Sample sizes: (normal, quick)
SAMPLE_SIZES: dict[str, tuple[int, int]] = {
    "legalbench": (24, 12),   # 4 (normal) or 2 (quick) per category × 6
    "cuad":       (10,  3),   # contracts per category
    "ifeval":     (100, 10),  # prompts
    "mmlupro":    (200, 20),  # law questions (200 ≈ full law split)
}

# ── LegalBench ─────────────────────────────────────────────────────────────────
LEGALBENCH_REPO_URL = "https://github.com/HazyResearch/legalbench"
LEGALBENCH_DIR      = DATASETS_DIR / "legalbench"

LEGALBENCH_CATEGORIES = [
    "issue-spotting",
    "rule-recall",
    "rule-conclusion",
    "rule-application",
    "interpretation",
    "rhetorical-understanding",
]

# ── CUAD ───────────────────────────────────────────────────────────────────────
CUAD_REPO_URL = "https://github.com/TheAtticusProject/cuad"
CUAD_DIR      = DATASETS_DIR / "cuad"

CUAD_IP_CATEGORIES = [
    "Change-of-Control",
    "Non-Compete",
    "Anti-Assignment",
    "Exclusivity",
    "Governing-Law",
    "Renewal-Term",
    "Expiration-Date",
    "Parties",
]

# ── IFEval ─────────────────────────────────────────────────────────────────────
IFEVAL_HF_DATASET = "google/IFEval"

# ── MMLU-Pro ───────────────────────────────────────────────────────────────────
MMLUPRO_HF_DATASET   = "TIGER-Lab/MMLU-Pro"
MMLUPRO_LAW_SUBJECTS = {"law", "jurisprudence"}

# ── Checkpoint ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = BENCHMARK_ROOT / "checkpoints"
