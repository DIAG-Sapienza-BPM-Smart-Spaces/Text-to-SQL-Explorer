## Repository overview

In this repository there is the code to both generate and visualize the data for the Demo.

Data are divided in two category: real execution data on bird_development dataset (excluding non-deepseek LLM selectors), and test_data available for generation.

Real execution data is already pre-calculated, while test-data is not. The small data selection direcltly available is due to size-issues. More real data is locally stored and is available on-demand.

We suggest to avoid recalculating real data, and instead generating the test-data (if wanted) for the full demo visualization.

## Setup

To run the code is necessary to install the libraries in the requirements.txt file.

langchain is actually needed only to generate real data for the LLM-as-judges, so it may be skipped if focusing only on the visualization.

We suggest installing the libraries in a conda virtual environment to avoid any issues.

```bash
conda create -n demo_paper python==3.14
```

```bash
conda activate demo_paper
```


```bash
pip install -r requirements.txt

```

```bash
conda install -n demo_paper -c conda-forge scikit-learn
```

## Test-data Generation

To generate test data for missing selectors and datasets, simply run:

```bash
python generate_test_visualization_data.py
```

**The generation process may be take a while given the size of the output data**

There is a Develpment flag at the start of both first_visualizion.py and binary_visualization.py that allow to turn off/on the use of test data


## Generation Data Pipeline Commands for Reproducibility

This are the step needed to generate the real data artifacts for visualization, given a set of model candidates.

**The generation process may be take a while given the size of the input data**

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

**Next steps require setting and apy_key for deepseek or similar models inside the selector.py and llm_as_judge.py files**

4. Generate pairwise selector judgments (models and datasets must be stated in the code):

```bash
python selector.py
```

5. Generate binary judges outputs (models and datasets must be stated in the code):

```bash
python llm_as_judge.py
```

## Visualization command

To run the visualization, simply run:

```bash
streamlit run combined_visualizion.py
```

A browser window with the visualizions should open.

**If using test data for all datasets, the visualization may be slightly congested. This is due to the huge amount of data being filtered and visualized.**

*There is a Develpment flag at the start of both first_visualizion.py and binary_visualization.py that allow to turn off/on the use of test data.*

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
