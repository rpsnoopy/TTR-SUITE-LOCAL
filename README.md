# TTR-SUITE-LOCAL

Benchmark suite per la selezione di un LLM locale adatto a TTR-SUITE (piattaforma di analisi IP/contratti).

## Obiettivo

Identificare il miglior LLM locale eseguibile su **RTX 4090 16GB VRAM** con performance paragonabili a Claude Sonnet 4.6 su task legali: analisi contratti, NDA, clausole IP.

## Risultati Finali (23 febbraio 2026)

| Modello | TTR-Score | LegalBench | CUAD | IFEval | MMLU-Pro | tok/s |
|---------|:---------:|:----------:|:----:|:------:|:--------:|:-----:|
| **qwen3:14b** ⭐ | **83.5%** | **95.8%** | 51.2% | 87.0% | 42.5% | **35** |
| Claude Sonnet 4.6 (API) | 81.6% | 91.7% | **53.8%** | **93.0%** | **62.0%** | — |
| mistral-small:24b | 77.9% | 91.7% | 47.5% | 80.0% | 39.5% | 21 |
| gpt-oss:20b ⚡ | 75.1% | 87.5% | 51.2% | 86.0% | 38.5% | **80** |
| deepcoder:14b | 74.4% | 83.3% | 46.2% | 79.0% | 43.0% | 32 |
| qwen3:30b-a3b | 70.9% | 79.2% | 48.8% | 84.0% | **48.0%** | 30 |
| phi4:14b | 65.7% | 83.3% | 3.8%¹ | 79.0% | 46.0% | 35 |
| qwen3:32b | — | — | 46.2% | — | — | 5⛔ |

> ¹ phi4:14b CUAD: rifiuta di estrarre clausole nel 96% dei casi → F1 quasi zero

**Raccomandazione:** `qwen3:14b` — miglior TTR-Score (83.5%), supera Claude 4.6 su LegalBench (95.8%), 9GB VRAM, 35 tok/s. Claude API per MMLU-Pro-class tasks (knowledge gap reale: 62% vs 48%).

## Struttura Repository

```
benchmark/
├── benchmark_runner.py     # Entry point: --quick, --dry-run, --resume, --no-pull
├── config.py               # Modelli, path, timeout, BENCHMARK_NUM_PREDICT
├── consolidate_results.py  # XLSX 9 fogli + dedup automatico per task_id
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

# Consolida tutti i risultati in XLSX (dedup automatico, ultimo run per task vince)
python consolidate_results.py --results-dir C:\TTR_Benchmark\results
```

> **Nota:** Il benchmark runner usa **Windows Ollama** (non WSL). Fare sempre pull dei modelli dalla GUI Windows o con `ollama.exe pull`.

---

*Aviolab AI — TTR-SUITE Benchmark Suite v1.2, febbraio 2026*
