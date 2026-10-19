<div align="center">

# CIVIC-CR

### Causal Intervention Verification for Critique Reliability in Code Review

**Does a review critique track the defect it claims to identify?**

[Method](#method) · [Code](#code) · [Getting started](#getting-started) · [Reproducibility](docs/REPRODUCIBILITY.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20code-8574B5)

</div>

![CIVIC-CR method overview](assets/overview.png)

CIVIC-CR studies verification of AI-generated code-review findings through **critique-conditioned program interventions**. A convincing explanation alone does not establish that a defect exists. We instead ask whether a finding responds to changes in the specific mechanism it alleges.

The repository contains research scripts for frozen embeddings, lightweight verification, intervention ranking, controlled evaluations, and downstream repair experiments. The figure illustrates the research design; individual scripts implement subsets of it. No conference acceptance, released paper, or fully reproduced result is claimed here.

## Method

For a program `x` and a candidate critique `c`, the verifier produces a validity score `sθ(x, c)`. Three complementary checks characterize the intended behavior:

| Check | Intervention | Expected response |
| :--- | :--- | :--- |
| Target sensitivity | Repair the defect claimed by `c` | The critique's validity decreases |
| Causal specificity | Repair an unrelated mechanism | The critique's validity stays stable |
| Structured equivariance | Apply a behavior-preserving transformation | Validity stays stable; location and evidence move consistently |

The key contrast is between `sθ(x,c) − sθ(I_c(x),c)` and the score change under an unrelated intervention. The reviewer, intervention generator, and embedding backbone remain frozen; the upper verifier and intervention ranker are trained.

A typical failure case is a null-dereference warning on code that already has a dominating null guard. A redundant edit should not become evidence that the original warning was correct. Intervention quality and executable validation are therefore part of the evaluation assumptions.

## Code

```text
civic/         Importable verifier models, intervention ranker, and workspace paths
scripts/       Training, evaluations, ablations, and repair experiments
tools/         Data preparation, embeddings, diagnostics, and service checks
env/           Dependency requirements and credential-free environment example
assets/        Method overview
docs/          Reproducibility and release notes
outputs/       Generated experiment artifacts (ignored by Git)
```

| Entry point | Purpose |
| :--- | :--- |
| `civic/models.py` | MLP, bilinear, and Transformer-based verifier definitions |
| `tools/build_emb_diff.py` | Build aligned code/critique embeddings from intervention data |
| `tools/build_finding_dataset.py`, `scripts/evaluate_finding_thresholds.py` | Construct finding-validity data and evaluate score thresholds |
| `scripts/train_civic.py` | Joint classification, sensitivity, specificity, and ranking training |
| `scripts/run_objective_ablations.py` | Five intervention-objective configurations across three seeds |
| `scripts/frozen_reviewer_baseline.py`, `scripts/evaluate_finding_baselines.py` | Frozen reviewer and finding-verification baselines |
| `scripts/evaluate_matched_pairs.py` | Discriminate defective code from its matched repaired counterpart |
| `scripts/train_specificity_ablation.py`, `scripts/train_score_battery_specificity.py` | Compare specificity objectives using embeddings or reviewer-score features |
| `scripts/evaluate_equivariance.py`, `scripts/evaluate_localization_equivariance.py` | Transformation and localization evaluations |
| `scripts/evaluate_cross_project_small.py` | Leave-project-out evaluation |
| `scripts/evaluate_reviewer_cross_language.py`, `scripts/evaluate_verifier_cross_language.py` | Cross-language evaluation of reviewer and verifier |
| `scripts/evaluate_ranking_baselines.py` | Intervention-ranking baselines |
| `scripts/spike_repair.py`, `scripts/summarize_downstream_repair.py` | Repair harness and downstream aggregation |

## Getting started

Use Python 3.10 or newer. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export CIVIC_PROJECT_ROOT="$PWD"
mkdir -p cache data/anchors repos_cache outputs
python -m scripts.train_civic --help
```

The training scripts require external benchmark data and precomputed artifacts. **This checkout does not include datasets, intervention quadruples, split manifests, embeddings, checkpoints, or experimental outputs.** Installation alone is insufficient to reproduce the experiments. See [the required inputs and known limitations](docs/REPRODUCIBILITY.md).

With those artifacts prepared:

```bash
python -m tools.build_emb_diff
python -m tools.embed_benchmarks --bench aacr
python -m scripts.train_civic --seed 42
python -m scripts.run_objective_ablations
```

### Optional model services

Embedding and verifier training can run locally with downloaded backbone weights. LLM-based construction and review scripts additionally require your own service credentials. The variable names are listed in [env/.env.example](env/.env.example); scripts read the process environment and do not automatically load that file. Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` for OpenAI-compatible calls, or `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` for Anthropic calls. Azure embedding experiments use their own endpoint and key variables.

Some historical scripts select fixed model identifiers or service-specific API methods. Check the selected script and your provider's support before a paid run. Repair experiments also require Docker, benchmark images, and suitable compute.

## Evaluation scope

The research questions cover false positives at matched defect recall, target-versus-unrelated intervention discrimination, transformation consistency, generalization, intervention ranking, and downstream repair utility. Pair AUROC measures intervention discrimination; it is not interchangeable with finding-validity AUROC.

Causal evidence depends on an intervention actually removing the alleged defect without introducing confounding changes. Subjective review comments and unfixable findings may not support this test. The repository retains exploratory and negative-result scripts; no performance table is published without accompanying artifacts.

## Release notes

See [release validation](docs/RELEASE_NOTES.md) for the checks performed and [reproducibility](docs/REPRODUCIBILITY.md) for implementation caveats. This repository currently has no license grant; redistribution and reuse permissions require clarification from the rights holder. Third-party benchmarks and model weights retain their own terms.
