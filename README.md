## Repository overview

This repository contains the code to run locally the demo of the Text-to-SQL Explorer paper.

With respect to the live demo, due to size constraints this repository includes only a subset of the real execution results on the BIRD development dataset (limited to DeepSeek-based LLM selectors); the complete dataset is available on request and will be provided for the live demo.

To reproduce all demo functionalities locally, apart from requesting the complete dataset, it is possible to:

- Generate synthetic test data locally

- Recompute all real data from scratch (time-consuming)

Both procedures are described below.

## Setup

To run the code is necessary to install the libraries in the requirements.txt file.

langchain is actually needed only to genereate real data for the LLM-as-judges, so it may be skipped if focusing on visualization.

We suggest installing the libraries in a conda virtual environment to avoid any issues.

```bash
conda create -n demo_paper python==3.14
```


```bash
pip install -r requirements.txt

```

## Test-data Generation

To generate test data for missing selectors and datasets, simply run:

```bash
python generate_test_visualization_data.py
```

There is a Develpment flag at the start of both first_visualizion.py and binary_visualization.py that allow to turn off/on the use of test data


## Visualization command

To run the visualization, simply run:

```bash
streamlit run combined_visualizion.py
```

A browser window with the visualizions should open.

There is a Develpment flag at the start of both first_visualizion.py and binary_visualization.py that allow to turn off/on the use of test data.


## Generation Data Pipeline Commands for Reproducibility

This are the step needed to generate the real data artifacts for visualization, given a set of model candidates.

Input models candidate SQL files are expected under candidates/ as \*\_query_results.json files.

1. Build canonical metrics lookup for real data metrics:

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

4. Generate pairwise selector judgments (models and datasets must be stated in the code):

```bash
python selector.py
```

5. Generate binary judges outputs (models and datasets must be stated in the code):

```bash
python llm_as_judge.py
```

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
- test_data/test_selector_pairwise_results.json
