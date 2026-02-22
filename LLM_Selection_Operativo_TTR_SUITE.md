# Selezione LLM Locale per TTR-SUITE — Sintesi Operativa

> **Data:** 22 febbraio 2026 (aggiornato con risultati benchmark v2 — prompt corretti)
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

### Nota metodologica — LegalBench v1 vs v2

I risultati LegalBench iniziali (v1, 8–37%) erano affetti da un problema di prompt: il runner usava un formato generico ("Task: nome\n\nTesto\n\nAnswer:") ignorando i `claude_prompt.txt` forniti da LegalBench con regole, tassonomia e few-shot examples specifici per ogni task. Una volta corretti i prompt (v2), tutti i modelli migliorano drasticamente (+54–83 pp). **I risultati v2 sono quelli corretti e confrontabili.**

### Tabella risultati (LegalBench v2 — prompt corretti)

| Modello | LegalBench | CUAD | IFEval | MMLU-Pro (Law) | Avg tok/s | VRAM |
|---------|-----------|------|--------|-----------------|-----------|------|
| **Claude Sonnet 4.6** (riferimento API) | 91.7% | **53.8%** | **93.0%** | **62.5%** | — API | — |
| **qwen3:14b** ⭐ | **95.8%** | 51.2% | 87.0% | 15.5% | **35** | ~9GB |
| **mistral-small:24b** | 91.7% | 51.2% | 81.0% | **37.0%** | 21 | ~14GB |
| **gpt-oss:20b** ⚡ | 87.5% | 51.2% | 86.0% | 23.5% | **80** | ~13GB |
| **deepcoder:14b** (DeepSeek-R1) | 83.3% | 46.2% | 79.0% | 23.2% | 32 | ~9GB |
| **qwen3:30b-a3b** | 79.2% | 48.8% | 84.0% | 11.0% | 30 | ~18GB* |
| **qwen3:32b** ⛔ | — | 46.2% | — | — | **5** | split CPU |

> *qwen3:30b-a3b → CPU/GPU split su 16GB, velocità degradata
> ⛔ qwen3:32b interrotto: 5 tok/s inaccettabile (CPU/GPU split su 16GB)

### Gap rispetto a Claude Sonnet 4.6 (v2)

| Benchmark | Miglior locale | Gap vs Claude |
|-----------|---------------|---------------|
| LegalBench | **qwen3:14b (95.8%)** | **+4.2 pp** ← supera Claude |
| CUAD | gpt-oss:20b / qwen3:14b / mistral-small (51.2%) | -2.6 pp |
| IFEval | qwen3:14b (87.0%) | -6.0 pp |
| MMLU-Pro | mistral-small (37.0%) | -25.5 pp |

**Osservazione chiave:** Con i prompt corretti, i modelli locali sono **competitivi o superiori** a Claude Sonnet 4.6 su LegalBench. Il gap LegalBench precedentemente osservato (8–21%) era un artefatto metodologico, non una limitazione intrinseca dei modelli. Il vero discriminante rimane **MMLU-Pro** (conoscenza giuridica profonda): Claude 62.5% vs mistral-small 37% — gap reale di 25.5 pp.

**CUAD e IFEval** rimangono i benchmark con minore dispersione tra i modelli: gap Claude–miglior locale di 2.6 pp (CUAD) e 6.0 pp (IFEval). I modelli locali sono sostanzialmente equivalenti a Claude su questi task.

---

## 4. Raccomandazione Aggiornata

### Strategia ibrida per TTR-SUITE (RTX 4090 16GB)

| Livello | Modello | Caso d'uso | tok/s |
|---------|---------|-----------|-------|
| **Quick scan / produzione** | `gpt-oss:20b` | Pre-screening, classificazione, estrazione strutturata — velocità massima | **80** |
| **Analisi legale** | `qwen3:14b` | LegalBench best local (95.8%), CUAD/IFEval competitivi, 9GB pure GPU | 35 |
| **Conoscenza giuridica** | `mistral-small:24b` | MMLU-Pro best local (37%), ragionamento giuridico multi-step | 21 |
| **Validazione finale** | Claude Sonnet 4.6 API | Casi ad alto rischio, MMLU-Pro (62.5% gap reale vs 37% local) | — |

### Raccomandazione modello singolo

Se è necessario un unico modello locale: **`qwen3:14b`**
- **Miglior LegalBench locale** (95.8% — supera Claude 91.7%)
- CUAD competitivo (51.2%, pari a gpt-oss e mistral-small)
- IFEval ottimo (87.0%)
- Solo 9GB → entra pure-GPU in 16GB con ampio margine
- 35 tok/s — veloce e compatto
- Punto debole: MMLU-Pro (15.5% vs 37% di mistral-small) — per task di conoscenza giuridica profonda usare mistral-small o Claude API

Se la priorità è la **velocità massima**: **`gpt-oss:20b`** (80 tok/s, LegalBench 87.5%, CUAD/IFEval competitivi).

Se la priorità è la **conoscenza giuridica** (MMLU-Pro): **`mistral-small:24b`** (37% MMLU-Pro, LegalBench 91.7%).

### Cosa non usare in produzione TTR-SUITE
- **qwen3:32b** — 5 tok/s su 16GB VRAM: inutilizzabile per uso interattivo
- **qwen3:30b-a3b** — CPU/GPU split degradato, velocità e qualità compromesse su 16GB; qwen3:14b è migliore su LegalBench e 3x più veloce
- **deepcoder:14b** — coder specializzato, non porta vantaggi su task legali rispetto a qwen3:14b (83.3% vs 95.8% LegalBench, CUAD/IFEval inferiori)

---

## 5. Rischi e Limitazioni

| Rischio | Stato | Mitigazione |
|---------|-------|-------------|
| LegalBench prompt fix cambia il ranking | **Risolto** | Tutti i modelli ri-testati con prompt corretti (v2) |
| qwen3:32b inutilizzabile su 16GB | **Confermato** | Escluso; usare qwen3:14b o gpt-oss:20b |
| CPU/GPU split su modelli >14GB | **Confermato** | Stare sotto 14GB per uso produttivo |
| MMLU-Pro gap (37% local vs 62.5% Claude) | **Confermato** | Per ragionamento giuridico complesso usare sempre Claude API |
| qwen3:14b MMLU-Pro solo 15.5% | **Confermato** | Combinare con mistral-small per task knowledge-intensive |
| LegalBench è su diritto USA, non italiano/europeo | Aperto | Creare task supplementari su diritto italiano/GDPR/AI Act |
| Benchmark accademici ≠ qualità output reale | Aperto | Integrare con test su documenti TTR-SUITE reali anonimizzati |
| gpt-oss:20b genera solo thinking tokens su alcuni task | Aperto | Aumentare num_predict o usare prompt più strutturati |

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

### Re-eseguire un benchmark con prompt aggiornati

Se si modifica `benchmarks/legalbench.py` (o altro benchmark), **ri-eseguire tutti i modelli** inclusi Claude API per mantenere la comparabilità:

```bash
# Tutti i locali
python benchmark_runner.py --models qwen3-14b qwen3-30b-a3b deepcoder-14b mistral-small-24b gpt-oss-20b --benchmarks legalbench --no-pull

# Claude API (richiede ANTHROPIC_API_KEY in env Windows)
python benchmark_runner.py --models claude-sonnet-4-6 --benchmarks legalbench
```

---

## 7. Modelli Pianificati / Da Testare

| Modello | VRAM est. | Note |
|---------|----------|------|
| `qwen2.5-coder:14b` | ~9GB | Confronto diretto con deepcoder:14b su task legali |
| `llama3.3:70b` | >16GB | Richiederebbe upgrade VRAM o offload |
| `phi-4:14b` | ~9GB | Microsoft Phi-4, forte su ragionamento |
| `gemma3:12b` | ~8GB | Google Gemma 3, potenzialmente forte su legal |

---

*Aviolab AI — TTR-SUITE Benchmark Suite v1.1, febbraio 2026*
