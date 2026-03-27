Scenario 1 — Identifying the Best Text-to-SQL Model with Ground Truth
In questo scenario l’utente può confrontare diversi modelli Text-to-SQL utilizzando query con ground truth disponibili.
Il sistema permette di selezionare una o più metriche di valutazione (ad esempio exact match, execution accuracy, ecc.) e restituisce un confronto tra i modelli, evidenziando il modello complessivamente migliore rispetto alla metrica scelta.
Questo scenario consente di:
analizzare le prestazioni comparative dei modelli,

comprendere come diverse metriche influenzano la valutazione,

identificare il miglior sistema per un determinato benchmark.
Scenario 2 — Agent-Based Selection of Text-to-SQL Predictions
In questo scenario la scelta della migliore query SQL non è effettuata direttamente dall’utente, ma da un agente automatico.
Sono possibili diverse strategie di selezione, ad esempio:
Judge-based ensemble: un agente giudice valuta le diverse query candidate e restituisce un verdetto di affidabilità (ad esempio trust / do not trust).

LLM-based reasoning: un LLM applica regole sintattiche e semantiche, tramite prompting, per selezionare la query SQL più plausibile.

L’obiettivo è dimostrare come meccanismi automatici di selezione o validazione possano supportare l’utente nella scelta della query corretta.
Scenario 3 — Investigating the Influence of Schema and Query Properties
Questo scenario permette di analizzare come le caratteristiche dello schema e delle query influenzano le prestazioni dei modelli Text-to-SQL.
L’utente può esplorare diversi fattori, ad esempio:
complessità della query (facile vs difficile),

lunghezza della query naturale,

numero di attributi o tabelle coinvolte,

caratteristiche dello schema del database.

Questo consente di studiare quando e perché alcuni modelli funzionano meglio di altri, offrendo una prospettiva più analitica sulle prestazioni.
Scenario 4 — Bring Your Own Data and Models
L’ultimo scenario consente agli utenti di utilizzare i propri dataset e modelli Text-to-SQL all’interno del sistema.
Attraverso un semplice meccanismo di upload, gli utenti possono:
caricare il proprio schema e le proprie query,

integrare modelli personalizzati,

eseguire le stesse analisi disponibili negli scenari precedenti.

Questo rende la demo generalizzabile e riutilizzabile anche su casi d’uso reali.

## Reworked Pipeline Commands

The reworked pipeline keeps the same goals but uses canonical metrics and precomputed artifacts.

Input candidate SQL files are expected under candidates/ as \*\_query_results.json files.

1. Build canonical metrics lookup for BIRD Developer true metrics:

```bash
python precompute_metrics_lookup.py
```

2. Precompute reusable SQL embeddings for all candidate models:

```bash
python precompute_embeddings.py
```

3. Precompute per-query similarity matrices and default threshold stats:

```bash
python precompute_similarity_matrices.py
```

4. Generate pairwise selector judgments (raw persisted outcomes):

```bash
python selector.py
```

5. Generate fake data for non-BIRD datasets (including fake pairwise selector artifacts):

```bash
python generate_fake_visualization_data.py
```


There is a Develpment flag at the start of both first_visualizion.py and binary_visualization.py that allow to turn off/on the use of fake data

## Canonical Metrics

- execution_accuracy
- exact_match
- sql_f1_score
- response_schema_f1_score
- cell_f1_score

## New Artifact Locations

- precomputed/embeddings/
- precomputed/similarity/
- precomputed/metrics/
- pairwise_results/
- fake_data/fake_selector_pairwise_results.json
