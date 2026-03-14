# Semantic Agent Routing

This repository contains code, data, and evaluation scripts for a semantic agent routing benchmark. The task is to map a user prompt to one or more valid specialized agents.

The project studies three routing methods:

- **KNN semantic routing** over agent profile embeddings
- **Multilabel ML routing** using a one-vs-rest linear SVM over prompt embeddings
- **WAR (Weighted Agent Routing)**, a LinUCB-based cost-aware policy layer on top of the ML candidate path

The final evaluation protocol uses **set-valued prediction**: each method predicts a dynamic-size set of agents per prompt and is evaluated with set-level metrics.

## Repository Contents

Core code:
- `main.py`
- `train_ml.py`
- `evaluate.py`
- `routers.py`
- `bandit.py`
- `agents.py`
- `dataset.py`
- `embedder.py`
- `vector_store.py`

Data:
- `prompts_gold_2000.csv`

Utilities:
- `make_plots.py`
- `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Main Training Command

```bash
python main.py --mode train --dataset prompts_gold_2000.csv --ml_out ml_router_multi_profile_seed42.pkl --split_by_template --ml_model svm --embedder_model all-mpnet-base-v2 --ml_mode multilabel --min_test_per_agent 10 --max_negatives_per_prompt 5 --split_seed 42
```

## Main Evaluation Command

```bash
python main.py --mode eval --dataset prompts_gold_2000.csv --ml_out ml_router_multi_profile_seed42.pkl --set_threshold 0.5 --alpha 1.0 --cost_lambda 0.02
```

## Ablations

Supported ablations:
- multilabel vs. pairwise routing
- with vs. without agent profile text

Example pairwise training:

```bash
python main.py --mode train --dataset prompts_gold_2000.csv --ml_out ml_router_pair_profile_seed42.pkl --split_by_template --ml_model svm --embedder_model all-mpnet-base-v2 --ml_mode pairwise --min_test_per_agent 10 --max_negatives_per_prompt 5 --split_seed 42
```

Example no-profile training:

```bash
python main.py --mode train --dataset prompts_gold_2000.csv --ml_out ml_router_multi_noprofile_seed42.pkl --split_by_template --ml_model svm --embedder_model all-mpnet-base-v2 --ml_mode multilabel --no_profile_text --min_test_per_agent 10 --max_negatives_per_prompt 5 --split_seed 42
```

## Metrics

Reported metrics are set-level metrics:
- Precision
- Recall
- F1
- Jaccard
- Exact Match
- Average predicted set size

These metrics are computed over predicted agent sets rather than single-label outputs.

## Reproducibility

The main reported setting uses:
- dataset: `prompts_gold_2000.csv`
- embedder: `all-mpnet-base-v2`
- threshold: `0.5`
- seeds: `42`, `414`, `4142`

## Notes

This repository focuses on the routing benchmark and evaluation pipeline. Deployment-level concepts such as route maps and agent meshes are treated as system-level abstractions rather than fully implemented runtime infrastructure in this codebase.
