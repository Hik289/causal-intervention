# CIVIC-CR

Python code for verifying code-review findings with program interventions. The package provides verifier models and intervention ranking, with scripts for data preparation, training, evaluation, ablations, and repair experiments.

## Repository layout

```text
civic/         Importable verifier models, intervention ranker, and workspace paths
scripts/       Training, evaluation, ablation, and repair entry points
tools/         Data preparation, embeddings, diagnostics, and service checks
env/           Dependency requirements and an empty credential template
```

## Installation

Use Python 3.10 or newer and run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export CIVIC_PROJECT_ROOT="$PWD"
mkdir -p cache data/anchors repos_cache outputs
python -m scripts.train_civic --help
```

`CIVIC_PROJECT_ROOT` selects the workspace containing data, caches, and outputs; it defaults to the current directory. Generated artifacts are ignored by Git.

## Entry points

| Entry point | Purpose |
| --- | --- |
| `civic/models.py` | MLP, bilinear, and Transformer verifier definitions |
| `civic/ranking.py` | Intervention ranker |
| `tools/build_emb_diff.py` | Build aligned code and critique embeddings |
| `tools/build_finding_dataset.py` | Construct finding-validity records |
| `scripts/train_civic.py` | Train a verifier and intervention ranker |
| `scripts/run_objective_ablations.py` | Run objective configurations across seeds |
| `scripts/evaluate_finding_baselines.py` | Evaluate finding-verification baselines |
| `scripts/evaluate_finding_thresholds.py` | Evaluate finding score thresholds |
| `scripts/evaluate_matched_pairs.py` | Evaluate defective and repaired code pairs |
| `scripts/evaluate_equivariance.py` | Evaluate transformation consistency |
| `scripts/evaluate_localization_equivariance.py` | Evaluate localization consistency |
| `scripts/evaluate_cross_project_small.py` | Run leave-project-out evaluation |
| `scripts/evaluate_reviewer_cross_language.py` | Evaluate reviewer transfer across languages |
| `scripts/evaluate_verifier_cross_language.py` | Evaluate verifier transfer across languages |
| `scripts/evaluate_ranking_baselines.py` | Evaluate intervention ranking |
| `scripts/spike_repair.py` | Run the repair harness |
| `scripts/summarize_downstream_repair.py` | Aggregate repair outputs |

Run entry points as `python -m scripts.<name>` or `python -m tools.<name>`.

## Required inputs

Datasets, intervention records, split manifests, embeddings, checkpoints, and experimental outputs must be prepared separately. The repository does not include a complete download and preparation pipeline.

| Input path | Consumer |
| --- | --- |
| `data/intervention_dataset/quadruples_final.jsonl` | Intervention embedding and transformation scripts |
| `data/swe-review-bench_raw/data/*.parquet` | Patch lookup and executable experiments |
| `cache/embfix_index.jsonl` | Existing aligned index and split assignments used by the diff builder |
| `cache/embdiff_arrays.npz` | Training arrays named `cx`, `cic`, `cicp`, and `crit` |
| `cache/embdiff_index.jsonl` | Aligned metadata with `quad_id`, `issue_id`, `repo`, and `split` |
| `data/aacr-bench_raw/dataset.json` | Benchmark embedding builder |
| `cache/bench_aacr_emb.npz` | Finding arrays named `code`, `crit`, and `labels` |
| `cache/bench_aacr_index.jsonl` | Finding metadata and split assignments |
| `repos_cache/` | Dedicated disposable benchmark clones |
| `data/anchors/` | Construction outputs used by repair and diagnostic scripts |
| `outputs/finding_dataset.jsonl` | Output of the finding builder and input to threshold evaluation |

Check the selected script for its required fields. The diff builder inherits existing split assignments; it does not create or validate the original project split.

## Optional model services

Local embedding and verifier runs require downloaded backbone weights. Scripts that call model services require your own credentials. [env/.env.example](env/.env.example) lists supported variables without credentials. Set variables in the process environment; scripts do not load this file automatically. Some scripts use fixed model identifiers or service-specific methods. Repair experiments also require Docker and benchmark images.

## Implementation notes

- `scripts/train_civic.py` trains the verifier and ranker. Its equivariance term is currently zero, and its minimality term uses frozen input differences without training the verifier.
- The main verifier attends over two projected embedding tokens rather than source-code tokens.
- In `scripts/run_objective_ablations.py`, the historical `clsonly` configuration is an untrained control; it does not optimize a classification loss.
- The diff builder includes diff context and reuses the original representation when an unrelated edit falls outside the selected file.
- Some experiment modules perform expensive work at import time. Run selected entry points directly.
- Checkout and patch scripts reset and clean their benchmark clones. Use dedicated disposable clones in `repos_cache/`.
- Dependencies are not fully pinned, and a validated end-to-end reproduction environment is not included.
