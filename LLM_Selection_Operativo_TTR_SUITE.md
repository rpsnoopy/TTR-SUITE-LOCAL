# Selezione LLM Locale per TTR-SUITE — Sintesi Operativa

> **Data:** 20 febbraio 2026  
> **Obiettivo:** Identificare un LLM locale eseguibile su RTX 3090 (24GB VRAM) con performance comparabili a Claude Sonnet 4.5 nell'analisi e sintesi di documenti legali/IP  
> **Deliverable:** Raccomandazione modello + piano benchmark + prompt Claude Code per esecuzione

---

## 1. Contesto e Vincolo Hardware

TTR-SUITE necessita di un LLM locale per scenari in cui l'uso di API cloud non è possibile o desiderabile (latenza, costi, privacy dati sensibili, operatività offline). Il vincolo è una singola NVIDIA RTX 3090 con 24GB VRAM GDDR6X (~936 GB/s bandwidth).

**Implicazioni del vincolo:**
- Modelli densi fino a ~32B parametri con quantizzazione Q4_K_M (~20GB modello)
- Modelli MoE fino a ~30B parametri totali (con pochi parametri attivi)
- Context window effettivo limitato: con 20GB usati dal modello, restano ~4GB per KV cache → ~4-8K token di contesto. Estendibile con KV cache quantizzata (q8_0 key + q4_0 value)
- Nessuno spazio per modelli 70B+ senza offload su RAM di sistema (prestazioni inaccettabili)

---

## 2. Benchmark Selezionati

Abbiamo identificato i benchmark in base a due criteri: **(a)** rilevanza per il dominio legale/IP e **(b)** disponibilità di dati comparativi per Sonnet 4.5 e i modelli candidati.

### Benchmark con dati già disponibili in letteratura

| Benchmark | Cosa misura | Perché serve per TTR-SUITE |
|-----------|------------|---------------------------|
| **MMLU-Pro** (sottocategoria Law) | Conoscenza giuridica e ragionamento multi-dominio | Verifica che il modello "sappia" abbastanza di diritto per non generare nonsense |
| **GPQA** | Ragionamento graduate-level su domande esperte | Proxy per la capacità di analizzare documenti tecnici complessi |
| **IFEval** | Fedeltà nel seguire istruzioni strutturate | Critico: gli agenti TTR-SUITE devono seguire procedure di analisi precise senza deviazioni |

### Benchmark da eseguire in proprio (dati non disponibili per modelli locali)

| Benchmark | Cosa misura | Perché serve per TTR-SUITE |
|-----------|------------|---------------------------|
| **LegalBench** (162 task) | Ragionamento legale: issue-spotting, rule-recall, interpretazione, retorica | Il più diretto — mappa esattamente le capacità richieste dagli agenti NDA/contratti |
| **CUAD** (41 categorie) | Contract Understanding: identificazione clausole IP, non-compete, indemnification | Direttamente allineato con l'analisi NDA e contratti commerciali di TTR-SUITE |

### Fonti dati per confronto con Sonnet 4.5

- **Vals.ai** (vals.ai/benchmarks): LegalBench, MMLU-Pro, GPQA per modelli cloud
- **EvalScope** (evalscope.readthedocs.io): Dati dettagliati Qwen3-32B thinking/non-thinking
- **LangDB** (langdb.ai/app/models/benchmarks): Tabella comparativa ~182 modelli
- **Qwen3 Technical Report** (arxiv.org/pdf/2505.09388): Dati ufficiali base models
- **Studio quantizzazione ionio.ai**: Degradazione <1% da Q4_K_M su GPQA/MMLU per Qwen2.5

---

## 3. Modelli Candidati e Performance Stimate

### Tabella comparativa principale

| Modello | MMLU-Pro | GPQA | IFEval (strict) | tok/s RTX 3090 | VRAM | Licenza |
|---------|---------|------|-----------------|----------------|------|---------|
| **Claude Sonnet 4.5** (riferimento) | ~82% | ~65% | ~87% | — (API) | — | Proprietary |
| **Qwen3-32B Thinking** ⭐ | 68.7% | 60.0% | 87.8% | ~25-30 | ~20GB | Apache 2.0 |
| **Qwen3-30B-A3B MoE** | ~55% | ~42% | ~65% | ~40-44 | ~18GB | Apache 2.0 |
| **Mistral Small 3.1 24B** | ~57% | ~40% | ~72% | ~45-55 | ~15GB | Apache 2.0 |

### Analisi dei gap rispetto a Sonnet 4.5

**Qwen3-32B Thinking** presenta un gap del 13-16% su MMLU-Pro e ~5% su GPQA rispetto a Sonnet 4.5. Su IFEval il gap è quasi nullo (~87.8% vs ~87%). Il gap reale sull'analisi legale non è noto — va misurato con LegalBench.

**Qwen3-30B-A3B** è significativamente più veloce (~44 tok/s vs ~25-30) ma con qualità inferiore (~55% MMLU-Pro). Utile per pre-screening o task meno critici.

**Mistral Small 3.1** è il più veloce tra i densi (~45-55 tok/s) e lascia più VRAM libera per context (~15GB → ~9GB per KV cache). Architettura con meno layer = minor latenza per forward pass.

### Note sulla velocità (tok/s su RTX 3090)

Le stime derivano da:
- Benchmark misurato: Qwen2.5-32B Q4_K_M → 34.23 tok/s (Dr. Daniel Bender, Ollama v0.3.2, Linux, monitor su GPU separata)
- Benchmark misurato: Qwen3-30B-A3B → 43.7 tok/s (keturk/llm_on_rtx_3090, Ubuntu)
- Benchmark RTX 5090: Mistral Small → 93 tok/s → stima 3090 ~50-60% → ~45-55 tok/s
- La RTX 3090 ha ~87-93% della bandwidth della 4090 → tok/s proporzionalmente inferiori
- **Windows 11 aggiunge ~5-10% overhead** rispetto a Linux per inference CUDA con Ollama
- **Thinking mode** genera token aggiuntivi (chain-of-thought) prima della risposta → la velocità percepita dall'utente è inferiore al tok/s raw

---

## 4. Raccomandazione

### Modello primario: Qwen3-32B (Thinking Mode) con Q4_K_M

**Motivazioni:**
1. Performance migliori nella categoria 24GB su tutti i benchmark cognitivi
2. Thinking mode produce chain-of-thought documentabile (utile per audit trail IP)
3. Quantizzazione Q4_K_M con degradazione <1% (studio ionio.ai su famiglia Qwen2.5)
4. Apache 2.0 — uso commerciale senza restrizioni
5. 119 lingue con supporto italiano nativo
6. Context 128K nativo (limitato dalla VRAM residua su 3090)

### Strategia ibrida a 3 livelli

| Livello | Modello | Uso | tok/s |
|---------|---------|-----|-------|
| **Quick scan** | Qwen3-30B-A3B | Pre-screening documenti, classificazione iniziale | ~40-44 |
| **Analisi** | Qwen3-32B Thinking | Analisi clausole, gap identification, sintesi | ~25-30 |
| **Validazione** | Sonnet 4.5 API | Casi ad alto rischio, revisione finale | — |

### Piano benchmark (prossimo passo)

Eseguire su Windows 11 + RTX 3090 con Ollama:

1. **LegalBench** — 162 task, open source (GitHub HazyResearch/legalbench) → dato proprietario confrontabile con Vals.ai
2. **CUAD** — 41 categorie clausole contrattuali → diretto per IP/NDA
3. **IFEval** — verifica instruction following → critico per agenti
4. **MMLU-Pro sottocategoria Law** → conferma conoscenza giuridica

Modelli da testare nel primo round:
- `qwen3:32b-q4_K_M` (thinking mode)
- `qwen3:30b-a3b` (Q4_K_M default)
- `mistral-small:24b` (Q4_K_M default)

---

## 5. Rischi e Limitazioni

| Rischio | Mitigazione |
|---------|------------|
| Context window effettivo ~4-8K con 32B su 3090 | Usare KV cache quantizzata; chunking documenti lunghi |
| Benchmark accademici non catturano qualità output legale reale | Integrare con test su documenti TTR-SUITE reali anonimizzati |
| Thinking mode genera token extra → latenza percepita maggiore | Per batch processing è accettabile; per interattivo usare no-think o MoE |
| Windows overhead su CUDA inference | Considerare WSL2+Ubuntu per migliorare ~5-10% |
| LegalBench è su diritto USA, non italiano/europeo | Creare task supplementari su diritto italiano/GDPR/AI Act |

---

## 6. File Prodotti

| File | Contenuto |
|------|-----------|
| `LLM_Benchmark_Comparison_TTR_SUITE.xlsx` | Tabella comparativa completa (4 fogli) |
| `LLM_Selection_Operativo_TTR_SUITE.md` | Questo documento |
| Prompt Claude Code (sezione 7 sotto) | Per setup ed esecuzione benchmark con Ollama |

---

## 7. Prompt per Claude Code — Setup ed Esecuzione Benchmark

Il prompt seguente è progettato per essere incollato direttamente in Claude Code. Genera l'intera infrastruttura di test su Windows 11 con Ollama.

---

```
CONTESTO:
Sto selezionando un LLM locale per TTR-SUITE (piattaforma di analisi IP/contratti).
Devo benchmarkare 3 modelli su RTX 3090 (24GB) Windows 11 con Ollama.
I modelli sono:
1. qwen3:32b-q4_K_M (con thinking mode ON)
2. qwen3:30b-a3b (default q4)
3. mistral-small:24b (default q4)

OBIETTIVO:
Creare uno script Python completo che:

A) SETUP (una tantum):
- Verifichi che Ollama sia installato e raggiungibile (localhost:11434)
- Scarichi i 3 modelli se non presenti (ollama pull)
- Crei la directory di lavoro C:\TTR_Benchmark\

B) BENCHMARK 1 — LegalBench:
- Cloni o scarichi il dataset LegalBench da https://github.com/HazyResearch/legalbench
- Selezioni un sottoinsieme rappresentativo di task (minimo 20 task, coprendo tutte e 6 le categorie: issue-spotting, rule-recall, rule-conclusion, rule-application, interpretation, rhetorical-understanding)
- Per ogni task, invii il prompt al modello via API Ollama (POST http://localhost:11434/api/chat)
- Registri: risposta, tempo di risposta, token generati, tok/s
- Confronti la risposta con il ground truth del dataset
- Calcoli accuracy per task e per categoria

C) BENCHMARK 2 — IFEval:
- Scarichi il dataset IFEval da Hugging Face (google/IFEval)
- Selezioni un campione di 50-100 prompt
- Invii al modello e valuti la conformità alle istruzioni (strict e loose)
- Registri metriche: prompt_level_strict_acc, inst_level_strict_acc

D) BENCHMARK 3 — CUAD:
- Scarichi il dataset CUAD da https://github.com/TheAtticusProject/cuad
- Selezioni le categorie più rilevanti per IP: IP Ownership Assignment, Non-Compete, License Grant, Limitation of Liability, Indemnification, Termination for Convenience, Change of Control, Audit Rights (8 categorie su 41)
- Per ogni clausola, chiedi al modello di identificare e estrarre la clausola rilevante dal contratto
- Confronta con le annotazioni ground truth
- Calcola F1-score per categoria

E) BENCHMARK 4 — MMLU-Pro Law Subset:
- Scarichi MMLU-Pro da Hugging Face
- Filtra solo le domande della categoria "Law" e "Jurisprudence"
- Invia come multiple-choice al modello
- Calcola accuracy

F) OUTPUT:
- Genera un file CSV con tutti i risultati raw: modello, benchmark, task, risposta, ground_truth, corretto, tempo_ms, token_count, tok_s
- Genera un file XLSX di riepilogo con:
  - Foglio 1: Accuracy per modello per benchmark
  - Foglio 2: Velocità media tok/s per modello
  - Foglio 3: Dettaglio per categoria LegalBench
  - Foglio 4: Dettaglio per categoria CUAD
- Salva tutto in C:\TTR_Benchmark\results\

VINCOLI TECNICI:
- Windows 11, PowerShell o CMD
- Python 3.11+, usa pip install per dipendenze
- Ollama API su localhost:11434
- RTX 3090 24GB, un solo modello alla volta in VRAM
- Tra un modello e l'altro, scarica il precedente (ollama stop) per liberare VRAM
- Per Qwen3-32B, abilita thinking mode (/think nel system prompt o enable_thinking=True)
- Timeout per singola richiesta: 120 secondi (i documenti CUAD sono lunghi)
- Se un benchmark richiede troppo tempo, implementa un --quick flag che riduce il campione a 10 task per benchmark

QUALITÀ CODICE:
- Codice production-ready con error handling
- Logging con timestamp in un file .log
- Possibilità di riprendere da dove si è interrotto (salva checkpoint dopo ogni task)
- Progress bar con tqdm
- Commenti in italiano
```

---

*Fine documento — Aviolab AI, febbraio 2026*
