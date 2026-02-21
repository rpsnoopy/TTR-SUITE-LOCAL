# TTR-SUITE Benchmark Suite — Quick Start

## Panoramica

Questo tool valuta LLM su 4 dataset legali (LegalBench, CUAD, IFEval, MMLU-Pro Law)
e produce un report CSV + XLSX con accuracy e throughput.

**Modelli supportati:**

| Nome CLI             | Dove gira        | Note                          |
|----------------------|------------------|-------------------------------|
| `qwen3-32b`          | Ollama (Windows) | Thinking mode attivo          |
| `qwen3-30b-a3b`      | Ollama (Windows) | MoE, più veloce               |
| `mistral-small-24b`  | Ollama (Windows) | Baseline                      |
| `claude-sonnet-4-5`  | Anthropic API    | Richiede API key              |
| `claude-sonnet-4-6`  | Anthropic API    | Richiede API key              |

---

## Ambienti: Windows PowerShell vs WSL

| Aspetto              | PowerShell (Win11)              | WSL (Ubuntu)                    |
|----------------------|---------------------------------|---------------------------------|
| **Raccomandato per** | Tutto (Ollama è qui)            | Solo sviluppo/debug             |
| **Ollama**           | Nativo, zero latenza            | Via `localhost:11434` (funziona)|
| **Path output**      | `C:\TTR_Benchmark\results\`     | `/mnt/c/TTR_Benchmark/results/` |
| **Venv**             | `.venv\Scripts\activate`        | `source .venv/bin/activate`     |
| **Python**           | `python`                        | `python3`                       |

> **Consiglio:** usa sempre **PowerShell** per eseguire i benchmark. Ollama gira su
> Windows e i path nativi Windows (`C:\`) funzionano direttamente.

---

## Setup (una-tantum)

### 1. Apri PowerShell nella cartella benchmark

In Esplora File, naviga in:
```
C:\Users\rpsno\OneDrive\Documents\GitHub\TTR-SUITE-LOCAL\benchmark
```
Click destro sullo sfondo → **"Apri nel terminale"**

### 2. Crea il virtualenv e installa le dipendenze

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Per i modelli Claude, installa anche il SDK Anthropic:
```powershell
pip install anthropic
```

### 3. Configura la API key di Anthropic (solo per Claude)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Per renderla permanente (una volta sola):
```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","sk-ant-...","User")
```

### 4. Verifica che Ollama sia in esecuzione

L'app Ollama deve essere avviata (icona nella tray) oppure:
```powershell
ollama serve
```

### 5. Smoke test

```powershell
python benchmark_runner.py --dry-run
```
Atteso: `6 passed, 0 failed`

---

## Scaricare i modelli Ollama (una-tantum)

```powershell
ollama pull qwen3:30b-a3b          # ~18 GB  — più veloce, inizia da qui
ollama pull qwen3:32b-q4_K_M       # ~20 GB  — candidato primario
ollama pull mistral-small:24b      # ~14 GB  — baseline
```

Verifica modelli disponibili:
```powershell
ollama list
```

---

## Eseguire i benchmark

### Test rapido (consigliato per la prima volta)

Testa un solo modello su LegalBench con campione ridotto (~12 domande, ~10 min):
```powershell
python benchmark_runner.py --benchmarks legalbench --models qwen3-30b-a3b --quick
```

### Run completo su tutti i benchmark

```powershell
python benchmark_runner.py --models qwen3-30b-a3b
```

### Confronto modelli locali + Claude

```powershell
python benchmark_runner.py --models qwen3-32b qwen3-30b-a3b claude-sonnet-4-6 --quick
```

### Solo Claude (senza Ollama)

```powershell
python benchmark_runner.py --models claude-sonnet-4-5 claude-sonnet-4-6 --no-pull --quick
```

### Benchmark specifici

```powershell
# Solo contratti IP
python benchmark_runner.py --benchmarks cuad --models qwen3-32b

# Solo instruction following
python benchmark_runner.py --benchmarks ifeval mmlupro --models qwen3-30b-a3b --quick
```

---

## Riprendere un run interrotto

Se interrompi con **Ctrl+C**, il checkpoint viene salvato automaticamente.
Per riprendere:

```powershell
python benchmark_runner.py --resume
```

Il tool mostra i checkpoint disponibili e chiede quale riprendere.

---

## Output e risultati

Tutto viene salvato in `C:\TTR_Benchmark\results\`:

```
C:\TTR_Benchmark\results\
├── raw_results_20260220_235813.csv    ← una riga per ogni domanda valutata
└── summary_20260220_235813.xlsx       ← 4 fogli di riepilogo
```

### Fogli XLSX

| Foglio       | Contenuto                                          |
|--------------|----------------------------------------------------|
| `Accuracy`   | % corrette per modello × benchmark                 |
| `Throughput` | tok/s medi (con/senza thinking mode)               |
| `LegalBench` | Accuracy per categoria (6 categorie)               |
| `CUAD`       | F1-score per categoria IP (8 categorie)            |

### Tabella riepilogativa (stampata a schermo a fine run)

```
Model                         LEGALBENCH    CUAD  IFEVAL  MMLUPRO
-----------------------------------------------------------------
qwen3-32b                          72.3%   68.1%   85.4%    71.2%
qwen3-30b-a3b                      61.2%   55.6%   67.8%    59.3%
mistral-small-24b                  58.7%   52.1%   71.2%    57.4%
claude-sonnet-4-6                  88.1%   81.3%   92.0%    85.6%
```

---

## Opzioni CLI complete

```
python benchmark_runner.py [OPTIONS]

--models        Modelli da testare (default: tutti e tre i locali)
--benchmarks    Benchmark da eseguire (default: tutti e quattro)
--quick         Campione ridotto (~10-12 task per benchmark)
--resume        Riprende da checkpoint esistente
--run-id        Specifica un run ID manuale (utile per riprendere senza prompt)
--output-dir    Directory output (default: C:\TTR_Benchmark\results\)
--no-pull       Salta ollama pull (modelli già presenti)
--dry-run       Test infrastruttura senza chiamate API
```

---

## Note su Claude API

- **Costo stimato** per un run `--quick` con Claude Sonnet 4.6:
  ~$1–3 (dipende dalla lunghezza dei prompt CUAD)
- **Costo stimato** per un run completo: ~$10–30
- Il throughput (tok/s) di Claude riflette la latenza API, non la GPU locale —
  non confrontabile direttamente con i modelli Ollama
- Il campo `thinking_tokens` rimane 0 per Claude (extended thinking non abilitato)

---

## Troubleshooting rapido

| Problema                          | Soluzione                                      |
|-----------------------------------|------------------------------------------------|
| `Ollama reachable: FAIL`          | Avvia l'app Ollama o esegui `ollama serve`     |
| `404 Not Found` per un modello    | `ollama pull <tag>` — modello non scaricato    |
| `ANTHROPIC_API_KEY` non trovata   | Imposta la variabile d'ambiente (vedi Setup 3) |
| `0/0 correct` nel summary         | Tutti i task sono falliti — controlla i log in `logs/` |
| Dataset non trovato               | Cancella la cartella `datasets/<nome>` e riprova |
