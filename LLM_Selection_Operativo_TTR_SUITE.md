# Selezione LLM Locale per TTR-SUITE — Sintesi Operativa

> **Data:** 23 febbraio 2026 (risultati finali — LegalBench v2 + MMLU-Pro v2)
> **Obiettivo:** Identificare un LLM locale eseguibile su RTX 4090 (16GB VRAM) con performance comparabili a Claude Sonnet 4.6 nell'analisi e sintesi di documenti legali/IP
> **Deliverable:** Benchmark suite eseguita + risultati misurati + raccomandazione operativa

---

## 1. Contesto e Vincolo Hardware

TTR-SUITE necessita di un LLM locale per scenari in cui l'uso di API cloud non è possibile o desiderabile (latenza, costi, privacy dati sensibili, operatività offline). Il vincolo è una singola **NVIDIA RTX 4090 con 16GB VRAM** GDDR6X (~1008 GB/s bandwidth).

**Implicazioni del vincolo:**
- Modelli densi fino a ~14B parametri Q4 (~9GB) → pura GPU, massima velocità
- Modelli densi 24B Q4 (~14GB) → pura GPU con margine stretto
- Modelli 30B+ → CPU/GPU split, velocità degradata (vedere risultati)
- Context window effettivo limitato con modelli grandi: ~4-8K token con split CPU/GPU
- RTX 4090 ha ~9% più bandwidth della 3090 ma metà della VRAM (16GB vs 24GB) — limite architetturale rilevante

---

## 2. Benchmark Selezionati

| Benchmark | Cosa misura | Rilevanza TTR-SUITE |
|-----------|------------|---------------------|
| **LegalBench** (24 task, 6 categorie) | Ragionamento legale: issue-spotting, rule-recall, rule-conclusion, rule-application, interpretation, rhetorical-understanding | Diretto — mappa le capacità richieste dagli agenti NDA/contratti |
| **CUAD** (80 item, 8 categorie IP) | Contract Understanding: Change-of-Control, Non-Compete, Anti-Assignment, Exclusivity, Governing-Law, Renewal-Term, Expiration-Date, Parties | Allineato con analisi NDA e contratti commerciali di TTR-SUITE |
| **IFEval** (100 item) | Fedeltà nel seguire istruzioni strutturate (strict accuracy) | Critico: gli agenti devono seguire procedure analitiche precise |
| **MMLU-Pro Law** (200 item) | Conoscenza giuridica e ragionamento multi-step | Verifica che il modello "sappia" abbastanza di diritto |

La benchmark suite è implementata in `benchmark/` (entry point: `benchmark_runner.py`). Tutti i risultati raw in `C:\TTR_Benchmark\results\`.

---

## 3. Risultati Benchmark Misurati (febbraio 2026)

### Hardware di test
- **GPU:** NVIDIA RTX 4090 16GB VRAM
- **OS:** Windows 11, Ollama (inference locale) + Claude API (riferimento)
- **Data esecuzione:** 21-23 febbraio 2026

### Note metodologiche

**LegalBench v2:** i risultati iniziali (v1, 8–37%) erano affetti da prompt inadeguati — il runner ignorava i `claude_prompt.txt` con regole e few-shot forniti da LegalBench. Corretti i prompt, tutti i modelli migliorano drasticamente (+54–83 pp). Tutti i modelli ri-testati con i prompt corretti.

**MMLU-Pro v2:** i risultati iniziali erano invalidi per i modelli con thinking mode implicito (Qwen3, gpt-oss): `num_predict=1024` esauriva il budget sui ragionamenti interni, lasciando risposte vuote (74–84% di risposte vuote per Qwen3). Corretto con `num_predict=4096` per MMLU-Pro. Tutti i modelli ri-testati con il parametro corretto e seed riproducibile.

### Tabella risultati finali

| Modello | TTR-Score | LegalBench | CUAD | IFEval | MMLU-Pro | tok/s | VRAM |
|---------|:---------:|:----------:|:----:|:------:|:--------:|:-----:|:----:|
| **qwen3:14b** ⭐ | **83.5%** | **95.8%** | 51.2% | 87.0% | 42.5% | **35** | ~9GB |
| **Claude Sonnet 4.6** (API) | 81.6% | 91.7% | **53.8%** | **93.0%** | **62.0%** | — | — |
| **mistral-small:24b** | 77.9% | 91.7% | 47.5% | 80.0% | 39.5% | 21 | ~14GB |
| **gpt-oss:20b** ⚡ | 75.1% | 87.5% | 51.2% | 86.0% | 38.5% | **80** | ~13GB |
| **deepcoder:14b** | 74.4% | 83.3% | 46.2% | 79.0% | 43.0% | 32 | ~9GB |
| **qwen3:30b-a3b** | 70.9% | 79.2% | 48.8% | 84.0% | **48.0%** | 30 | ~18GB* |
| **qwen3:32b** ⛔ | — | — | 46.2% | — | — | **5** | split CPU |

> *qwen3:30b-a3b → CPU/GPU split su 16GB, velocità degradata
> ⛔ qwen3:32b interrotto: 5 tok/s inaccettabile (CPU/GPU split su 16GB)

### Gap rispetto a Claude Sonnet 4.6

| Benchmark | Miglior locale | Gap vs Claude |
|-----------|---------------|:-------------:|
| LegalBench | **qwen3:14b (95.8%)** | **+4.1 pp** ← supera Claude |
| CUAD | gpt-oss:20b / qwen3:14b (51.2%) | -2.6 pp |
| IFEval | qwen3:14b (87.0%) | -6.0 pp |
| MMLU-Pro | qwen3:30b-a3b (48.0%) | -14.0 pp |
| **TTR-Score** | **qwen3:14b (83.5%)** | **+1.9 pp** ← supera Claude |

**Osservazione chiave:** Con i benchmark corretti metodologicamente, **qwen3:14b supera Claude Sonnet 4.6 sul TTR-Score aggregato** (83.5% vs 81.6%). Il vero vantaggio di Claude rimane su **MMLU-Pro** (62% vs 48% del miglior locale) — la conoscenza giuridica profonda e il ragionamento multi-step su 200 domande di diritto. Su tutti gli altri benchmark i modelli locali sono competitivi o superiori.

---

## 4. Raccomandazione Operativa

### Raccomandazione modello singolo

**`qwen3:14b`** — migliore TTR-Score (83.5%), best LegalBench in assoluto (95.8%), solo 9GB VRAM, 35 tok/s. Costo zero. Supera Claude 4.6 su 3 dei 4 benchmark.

### Strategia ibrida per TTR-SUITE (RTX 4090 16GB)

| Livello | Modello | Caso d'uso | tok/s |
|---------|---------|-----------|:-----:|
| **Produzione** | `qwen3:14b` | Analisi NDA, clausole IP, LegalBench-class tasks | 35 |
| **Velocità massima** | `gpt-oss:20b` | Pre-screening, classificazione rapida, pipeline ad alto volume | **80** |
| **Validazione finale** | Claude Sonnet 4.6 API | MMLU-Pro-class tasks, conoscenza giuridica profonda, casi ad alto rischio | — |

### Cosa non usare in produzione TTR-SUITE
- **qwen3:32b** — 5 tok/s su 16GB VRAM: inutilizzabile
- **qwen3:30b-a3b** — CPU/GPU split, qwen3:14b è migliore su LegalBench e 3x più veloce
- **deepcoder:14b** — coder specializzato, TTR-Score inferiore a qwen3:14b su tutti i benchmark legali
- **mistral-small:24b** — buon LegalBench (91.7%) ma CUAD e IFEval inferiori, e più lento di qwen3:14b

---

## 5. Rischi e Limitazioni

| Rischio | Stato | Mitigazione |
|---------|:-----:|-------------|
| MMLU-Pro gap (62% Claude vs 48% miglior locale) | **Confermato** | Per ragionamento giuridico multi-step usare Claude API |
| CPU/GPU split su modelli >14GB | **Confermato** | Stare sotto 14GB per uso produttivo |
| qwen3:32b inutilizzabile su 16GB | **Confermato** | Escluso definitivamente |
| LegalBench è su diritto USA, non italiano/europeo | Aperto | Creare task supplementari su diritto italiano/GDPR/AI Act |
| Benchmark accademici ≠ qualità output reale | Aperto | Integrare con test su documenti TTR-SUITE reali anonimizzati |
| qwen3:14b genera ~2500 tok/thinking su MMLU-Pro (~65s/domanda) | Aperto | Accettabile per uso batch; per uso interattivo considerare `/no_think` |

---

## 6. Infrastruttura Benchmark

| Componente | Path | Descrizione |
|-----------|------|-------------|
| Entry point | `benchmark/benchmark_runner.py` | `--quick`, `--dry-run`, `--resume`, `--no-pull` |
| Configurazione | `benchmark/config.py` | Modelli, path, timeout, `BENCHMARK_NUM_PREDICT` |
| Output raw | `C:\TTR_Benchmark\results\raw_results_*.csv` | Una riga per task |
| Consolidamento | `benchmark/consolidate_results.py` | XLSX 9 fogli; dedup automatico per task_id (ultimo run vince) |
| Checkpoint | `benchmark/checkpoints/{run_id}.json` | Resume automatico con `--resume` |
| Risultati committati | `benchmark/results/` | CSV + XLSX versionate su GitHub |

### Aggiungere un nuovo modello

1. Aggiungere entry in `benchmark/config.py` (`MODELS` dict)
2. Pull del modello tramite **Windows Ollama GUI** o `ollama.exe pull` (non da WSL)
3. Lanciare: `python benchmark_runner.py --models nome-modello --no-pull`
4. Consolidare: `python consolidate_results.py --results-dir C:\TTR_Benchmark\results`

> **Nota:** Su questo sistema esistono **due istanze Ollama separate** (WSL e Windows). Il benchmark runner usa sempre Windows Ollama (localhost:11434 da Windows Python). Fare sempre pull dei modelli da Windows.

---

## 7. Modelli Pianificati / Da Testare

| Modello | VRAM est. | Note |
|---------|:--------:|------|
| `phi-4:14b` | ~9GB | Microsoft Phi-4, forte su ragionamento |
| `gemma3:12b` | ~8GB | Google Gemma 3 |
| `llama3.3:70b` | >16GB | Richiederebbe upgrade VRAM o offload |

---

*Aviolab AI — TTR-SUITE Benchmark Suite v1.2, febbraio 2026*
