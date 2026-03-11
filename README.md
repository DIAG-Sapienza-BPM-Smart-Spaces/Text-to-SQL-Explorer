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

