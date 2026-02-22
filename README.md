# TTR-SUITE-LOCAL

Benchmark suite per la selezione di un LLM locale adatto a TTR-SUITE (piattaforma di analisi IP/contratti).

## Obiettivo

Identificare il miglior LLM locale eseguibile su **RTX 4090 16GB VRAM** con performance paragonabili a Claude Sonnet 4.6 su task legali: analisi contratti, NDA, clausole IP.

## Risultati (febbraio 2026)

| Modello | LegalBench | CUAD | IFEval | MMLU-Pro | tok/s |
|---------|-----------|------|--------|----------|-------|
| Claude Sonnet 4.6 (API) | **37.5%** | **53.8%** | **93.0%** | **62.5%** | — |
| gpt-oss:20b | 16.7% | 51.2% | 86.0% | 23.5% | **80** |
| qwen3:30b-a3b | **20.8%** | 48.8% | 84.0% | 11.0% | 32 |
| qwen3:14b | 12.5% | 51.2% | 87.0% | 15.5% | 35 |
| deepcoder:14b | 12.5% | 46.2% | 79.0% | 23.2% | 32 |
| mistral-small:24b | 8.3% | 51.2% | 81.0% | **37.0%** | 21 |
| qwen3:32b | 16.7% | 46.2% | — | — | 5⛔ |

**Raccomandazione:** `gpt-oss:20b` come modello primario locale (miglior bilanciamento velocità/qualità), `mistral-small:24b` per task a priorità MMLU-Pro, Claude API per validazione finale.

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
python benchmark_runner.py --models gpt-oss-20b --no-pull

# Quick test (campione ridotto)
python benchmark_runner.py --models gpt-oss-20b --no-pull --quick

# Resume da checkpoint
python benchmark_runner.py --models deepcoder-14b --no-pull --run-id 20260222_132042 --resume

# Consolida tutti i risultati in XLSX
python consolidate_results.py --results-dir /mnt/c/TTR_Benchmark/results
```

> **Nota:** Il benchmark runner usa **Windows Ollama** (non WSL). Fare sempre pull dei modelli dalla GUI Windows o con `ollama.exe pull`.

---

*Aviolab AI — TTR-SUITE Benchmark Suite v1.0*
