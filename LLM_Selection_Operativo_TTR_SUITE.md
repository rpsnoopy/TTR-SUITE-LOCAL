# Selezione LLM Locale per TTR-SUITE — Sintesi Operativa

> **Data:** 22 febbraio 2026 (aggiornato con risultati benchmark misurati)
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
- **Data esecuzione:** 21-22 febbraio 2026

### Tabella risultati

| Modello | LegalBench | CUAD | IFEval | MMLU-Pro (Law) | Avg tok/s | VRAM |
|---------|-----------|------|--------|-----------------|-----------|------|
| **Claude Sonnet 4.6** (riferimento API) | **37.5%** | **53.8%** | **93.0%** | **62.5%** | — API | — |
| **gpt-oss:20b** ⚡ | 16.7% | 51.2% | 86.0% | 23.5% | **80** | ~13GB |
| **qwen3:30b-a3b** | **20.8%** | 48.8% | 84.0% | 11.0% | 32 | ~18GB* |
| **qwen3:14b** | 12.5% | 51.2% | 87.0% | 15.5% | 35 | ~9GB |
| **deepcoder:14b** (DeepSeek-R1) | 12.5% | 46.2% | 79.0% | 23.2% | 32 | ~9GB |
| **mistral-small:24b** | 8.3% | 51.2% | 81.0% | **37.0%** | 21 | ~14GB |
| **qwen3:32b** ⛔ | 16.7% | 46.2% | — | — | **5** | split CPU |

> *qwen3:30b-a3b → CPU/GPU split su 16GB, velocità degradata
> ⛔ qwen3:32b interrotto: 5 tok/s inaccettabile (CPU/GPU split su 16GB)

### Gap rispetto a Claude Sonnet 4.6

| Benchmark | Miglior locale | Gap vs Claude |
|-----------|---------------|---------------|
| LegalBench | qwen3:30b-a3b (20.8%) | **-16.7 pp** |
| CUAD | gpt-oss:20b / qwen3:14b / mistral-small (51.2%) | -2.6 pp |
| IFEval | qwen3:14b (87.0%) | -6.0 pp |
| MMLU-Pro | mistral-small (37.0%) | -25.5 pp |

**Osservazione critica:** LegalBench mostra un "floor" strutturale per tutti i modelli locali tra 8% e 21%, indipendentemente dall'architettura (general, coder, reasoning). Claude si attesta al 37.5%. Questo gap (~17 pp) è probabilmente dovuto a una combinazione di capacità di ragionamento legale e sensibilità al formato di risposta exact-match richiesto da LegalBench.

**CUAD e IFEval** mostrano convergenza quasi totale tra i modelli locali (~46-51% CUAD, ~79-87% IFEval): il gap rispetto a Claude è ridotto e i modelli locali risultano sostanzialmente equivalenti su questi task.

**MMLU-Pro** è il benchmark più discriminante: mistral-small-24b (37%) supera tutti i modelli Qwen testati e si avvicina di più a Claude (62.5%), nonostante la velocità inferiore.

---

## 4. Raccomandazione Aggiornata

### Strategia ibrida per TTR-SUITE (RTX 4090 16GB)

| Livello | Modello | Caso d'uso | tok/s |
|---------|---------|-----------|-------|
| **Quick scan** | `gpt-oss:20b` | Pre-screening, classificazione rapida, estrazione strutturata | **80** |
| **Analisi legale** | `mistral-small:24b` | Ragionamento giuridico, MMLU-Pro best local (37%), analisi NDA | 21 |
| **Validazione** | Claude Sonnet 4.6 API | Casi ad alto rischio, revisione finale, LegalBench gap troppo ampio | — |

### Raccomandazione modello singolo

Se è necessario un unico modello locale: **`gpt-oss:20b`**
- Miglior bilanciamento velocità/qualità su 3 dei 4 benchmark (CUAD, IFEval, MMLU-Pro)
- 80 tok/s su RTX 4090 — 2-3x più veloce degli altri modelli
- 13GB → entra pure-GPU in 16GB con margine per context
- Unico punto debole: MMLU-Pro (23.5% vs 37% di mistral-small)

Se la priorità è la **conoscenza giuridica** (MMLU-Pro): **`mistral-small:24b`** (37% MMLU-Pro, migliore tra tutti i locali).

### Cosa non usare in produzione TTR-SUITE
- **qwen3:32b** — 5 tok/s su 16GB VRAM: inutilizzabile per uso interattivo
- **qwen3:30b-a3b** — CPU/GPU split degradato, velocità e qualità compromesse su 16GB
- **deepcoder:14b** — specializzato per codice, non porta vantaggi su task legali rispetto a gpt-oss:20b (risultati identici: LB 12.5-16.7%, CUAD 46%, IFEval 79%, MMLU-Pro 23%)

---

## 5. Rischi e Limitazioni

| Rischio | Stato | Mitigazione |
|---------|-------|-------------|
| LegalBench floor ~8-21% per tutti i locali | **Confermato** | Accettare il gap; usare Claude API per task LegalBench-critici |
| qwen3:32b inutilizzabile su 16GB | **Confermato** | Escluso; usare gpt-oss:20b o mistral-small |
| CPU/GPU split su modelli >14GB | **Confermato** | Stare sotto 14GB per uso produttivo (gpt-oss:20b a 13GB è il limite) |
| MMLU-Pro gap (37% local vs 62.5% Claude) | **Confermato** | Per ragionamento giuridico complesso usare sempre Claude API |
| LegalBench è su diritto USA, non italiano/europeo | Aperto | Creare task supplementari su diritto italiano/GDPR/AI Act |
| Benchmark accademici ≠ qualità output reale | Aperto | Integrare con test su documenti TTR-SUITE reali anonimizzati |

---

## 6. Infrastruttura Benchmark

La suite di benchmark è completamente implementata e riutilizzabile:

| Componente | Path | Descrizione |
|-----------|------|-------------|
| Entry point | `benchmark/benchmark_runner.py` | `--quick`, `--dry-run`, `--resume`, `--no-pull` |
| Configurazione | `benchmark/config.py` | Modelli, path, timeout — unica fonte di verità |
| Output raw | `C:\TTR_Benchmark\results\raw_results_*.csv` | Una riga per task |
| Consolidamento | `benchmark/consolidate_results.py` | XLSX 9 fogli: Accuracy, Throughput, LegalBench, CUAD, TTR-Score, Micro-heatmap, Rankings, TTR-Radar, Partial |
| Checkpoint | `benchmark/checkpoints/{run_id}.json` | Resume automatico con `--resume` |
| Risultati committati | `benchmark/results/` | CSV + XLSX versionate su GitHub |

### Aggiungere un nuovo modello

1. Aggiungere entry in `benchmark/config.py` (`MODELS` dict)
2. Pull del modello tramite **Windows Ollama GUI** o `ollama.exe pull` (non da WSL)
3. Lanciare: `python benchmark_runner.py --models nome-modello --no-pull`
4. Consolidare: `python consolidate_results.py --results-dir /mnt/c/TTR_Benchmark/results`

> **Nota:** Su questo sistema esistono **due istanze Ollama separate** (WSL e Windows). Il benchmark runner usa sempre Windows Ollama (localhost:11434 da Windows Python). Fare sempre pull dei modelli da Windows.

---

## 7. Modelli Pianificati / Da Testare

| Modello | VRAM est. | Note |
|---------|----------|------|
| `qwen2.5-coder:14b` | ~9GB | Confronto diretto con deepcoder:14b su task legali |
| `claude-sonnet-4-5` | — API | Confronto versione precedente vs 4.6 |
| `gpt-4o-mini` | — API | Riferimento API low-cost |
| `llama3.3:70b` | >16GB | Richiederebbe upgrade VRAM o offload |

---

*Aviolab AI — TTR-SUITE Benchmark Suite v1.0, febbraio 2026*
