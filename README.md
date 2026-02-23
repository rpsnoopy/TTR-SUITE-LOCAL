# TTR-SUITE-LOCAL

Benchmark suite per la selezione di un LLM locale adatto a TTR-SUITE (piattaforma di analisi IP/contratti).

## Obiettivo

Identificare il miglior LLM locale eseguibile su **RTX 4090 16GB VRAM** con performance paragonabili a Claude Sonnet 4.6 su task legali: analisi contratti, NDA, clausole IP.

## Risultati Finali v1.4 (23 febbraio 2026) — CUAD seed=42, num_predict=1024

| Modello | TTR-Score | LegalBench | CUAD | IFEval | MMLU-Pro | tok/s |
|---------|:---------:|:----------:|:----:|:------:|:--------:|:-----:|
| **qwen3:14b** ⭐ | **82.5%** | **95.8%** | 45.0% | 87.0% | 42.5% | **35** |
| Claude Sonnet 4.6 (API) | 81.6% | 91.7% | **53.8%** | **93.0%** | **62.0%** | — |
| mistral-small:24b | 77.5% | 91.7% | 45.0% | 80.0% | 39.5% | 21 |
| gpt-oss:20b ⚡ | 73.7% | 87.5% | 42.5% | 86.0% | 38.5% | **80** |
| deepcoder:14b | 73.6% | 83.3% | 41.2% | 79.0% | 43.0% | 32 |
| qwen3:30b-a3b | 70.1% | 79.2% | 43.8%² | 84.0% | **48.0%** | 30 |
| phi4:14b | 65.5% | 83.3% | 2.5%¹ | 79.0% | 46.0% | 35 |
| qwen3:32b | — | — | — | — | — | 5⛔ |

> ¹ phi4:14b CUAD: aggiunge testo esplicativo dopo "NESSUNA CLAUSOLA PRESENTE" → precision penalty → F1≈0. Con formato rispettato: ~37–38%
> ² qwen3:30b-a3b CUAD: era 18.8% con num_predict=512 (thinking chains esaurivano il budget, 50/80 risposte vuote) → 43.8% con num_predict=1024
> CUAD v1.4: seed=42 fisso in cuad.py → tutti i modelli valutati sugli stessi 80 item identici

**Raccomandazione:** `qwen3:14b` — miglior TTR-Score (82.5%), supera Claude 4.6 su LegalBench (95.8%), 9GB VRAM, 35 tok/s. Claude API per CUAD avanzato e MMLU-Pro-class tasks.

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

*Aviolab AI — TTR-SUITE Benchmark Suite v1.4, febbraio 2026*
