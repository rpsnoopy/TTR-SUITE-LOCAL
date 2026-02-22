# TTR-SUITE-LOCAL

Benchmark suite per la selezione di un LLM locale adatto a TTR-SUITE (piattaforma di analisi IP/contratti).

## Obiettivo

Identificare il miglior LLM locale eseguibile su **RTX 4090 16GB VRAM** con performance paragonabili a Claude Sonnet 4.6 su task legali: analisi contratti, NDA, clausole IP.

## Risultati (febbraio 2026 — LegalBench v2, prompt corretti)

| Modello | LegalBench | CUAD | IFEval | MMLU-Pro | tok/s |
|---------|-----------|------|--------|----------|-------|
| Claude Sonnet 4.6 (API) | 91.7% | **53.8%** | **93.0%** | **62.5%** | — |
| qwen3:14b ⭐ | **95.8%** | 51.2% | 87.0% | 15.5% | **35** |
| mistral-small:24b | 91.7% | 51.2% | 81.0% | **37.0%** | 21 |
| gpt-oss:20b ⚡ | 87.5% | 51.2% | 86.0% | 23.5% | **80** |
| deepcoder:14b | 83.3% | 46.2% | 79.0% | 23.2% | 32 |
| qwen3:30b-a3b | 79.2% | 48.8% | 84.0% | 11.0% | 30 |
| qwen3:32b | — | 46.2% | — | — | 5⛔ |

> **Nota:** I risultati LegalBench v1 (8–37%) erano affetti da prompt inadeguati — vedi `LLM_Selection_Operativo_TTR_SUITE.md` §3.

**Raccomandazione:**
- `qwen3:14b` — miglior LegalBench locale (95.8%), solo 9GB, 35 tok/s
- `gpt-oss:20b` — velocità massima (80 tok/s), LegalBench 87.5%
- `mistral-small:24b` — miglior conoscenza giuridica MMLU-Pro (37%)
- Claude API — validazione finale, MMLU-Pro best (62.5%)

## Struttura Repository

```
benchmark/
├── benchmark_runner.py     # Entry point: --quick, --dry-run, --resume, --no-pull
├── config.py               # Modelli, path, timeout — unica fonte di verità
├── consolidate_results.py  # Genera XLSX 9 fogli da tutti i CSV
├── results/                # CSV raw + XLSX consolidati (versionati)
├── checkpoints/            # Checkpoint per resume automatico
├── datasets/               # LegalBench (git clone), CUAD/IFEval/MMLU-Pro (HuggingFace)
└── src/                    # OllamaClient, AnthropicClient, output, checkpoint

LLM_Selection_Operativo_TTR_SUITE.md   # Analisi completa e raccomandazioni
```

## Uso Rapido

```bash
cd benchmark

# Benchmark completo su un modello
python benchmark_runner.py --models qwen3-14b --no-pull

# Quick test (campione ridotto)
python benchmark_runner.py --models qwen3-14b --no-pull --quick

# Resume da checkpoint
python benchmark_runner.py --models deepcoder-14b --no-pull --run-id 20260222_132042 --resume

# Consolida tutti i risultati in XLSX
python consolidate_results.py --results-dir /mnt/c/TTR_Benchmark/results

# Re-run LegalBench su tutti i modelli (es. dopo aggiornamento prompt)
python benchmark_runner.py --models qwen3-14b qwen3-30b-a3b deepcoder-14b mistral-small-24b gpt-oss-20b --benchmarks legalbench --no-pull
```

> **Nota:** Il benchmark runner usa **Windows Ollama** (non WSL). Fare sempre pull dei modelli dalla GUI Windows o con `ollama.exe pull`.

---

*Aviolab AI — TTR-SUITE Benchmark Suite v1.1, febbraio 2026*
