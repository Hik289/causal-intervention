# CIVER

**Causal Intervention Verification for Reliable Code Review**

[![Python](https://img.shields.io/badge/Python-code-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-training-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

CIVER verifies a code-review critique by examining what changes when its claimed defect is repaired. It compares targeted interventions with unrelated modifications and semantics-preserving transformations, then selects valid, nonredundant findings.

![CIVER framework](assets/civic_pipeline.png)

## Framework

Three lightweight modules are jointly optimized over frozen code, critique, and code–critique representations.

| Module | Parameters | Role |
| --- | --- | --- |
| Critique module | $F_\theta$ | Score the same critique across program states with shared cross-attention and a scoring MLP. |
| Intervention module | $R_\phi$ | Estimate candidate relevance and select a defect-related intervention. |
| Verifier module | $G_\psi$ | Combine the original score with intervention responses to estimate validity and select a critique subset. |

The reviewer, representation encoders, and intervention generator remain frozen. Training uses relevance-weighted candidate responses so that gradients reach all three modules; inference selects the highest-ranked intervention. Token features additionally support location and supporting-context predictions. Pooled features represent one context position and do not provide source-line localization.

## Training objectives

| Objective | Purpose |
| --- | --- |
| Critique validity | Supervise the verifier's final validity probability. |
| Defect removal | Reduce the critique score after a verified targeted repair. |
| Unrelated modification | Preserve the score under changes unrelated to the claim. |
| Semantic transformation | Preserve the score across supplied semantics-preserving transformations. |
| Intervention ranking | Assign higher relevance to verified targeted repairs. |
| Intervention minimality | Penalize representation-space edit distance through the candidate relevance distribution. |
| Review selection | Optimize validity minus pairwise redundancy within each review. |

The modules expose separate parameter groups for joint optimization. Loss weights and training settings are configured in the training code and command-line interface. Exact subset selection is used by default; larger groups require an explicit approximation choice.

## Getting started

Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[test]"
python -m scripts.train_joint --help
```

After preparing the frozen feature arrays:

```bash
python -m scripts.train_joint \
  --data data/joint_features.npz \
  --output outputs/joint
```

The trainer keeps complete review groups together, chooses its checkpoint using validation loss, and evaluates the test split after loading that checkpoint. It writes the checkpoint, objective history, split sizes, and evaluation metrics to the output directory.

## Data interface

The joint trainer accepts an NPZ archive with the following arrays:

| Arrays | Content |
| --- | --- |
| `original`, `critique` | Frozen code and critique features. |
| `candidates` | Candidate targeted-intervention features for each critique. |
| `unrelated`, `transformed` | Corresponding control states, shared or supplied per candidate. |
| `pair_original`, `pair_candidates`, `pair_unrelated`, `pair_transformed` | Separate frozen code–critique representations for every program state. |
| `labels` | Critique-validity labels. |
| `target_index` | Verified target-candidate index, or `-1` when unannotated. |
| `group_id` | Review membership for nonredundant set selection. |
| `split` | Prespecified `train`, `val`, and optional `test` assignments. |
| `candidate_mask` | Optional mask for valid candidate entries. |

Code, critique, and paired representations may be pooled vectors or fixed-length token features. Supply valid token positions without padding. All critiques from a review must belong to the same split. The default joint path requires all four paired-state arrays; the interface offers an explicit unpaired feature ablation.

Transformed states must be supplied rather than replaced by a zero loss. Identical original/control representations are reported as data diagnostics and cannot establish a response to a distinct intervention.

## Repository layout

```text
civic/       Trainable modules, joint objectives, paths, and optional service adapters
scripts/     Joint training, evaluation, ablations, and executable repair experiments
tools/       Dataset construction, frozen features, and diagnostics
env/         Dependencies and an empty local configuration template
assets/      Framework figure
tests/       Offline checks of gradients, objectives, selection, and training
```

| Entry point | Purpose |
| --- | --- |
| `civic/models.py` | Shared critique scorer, response verifier, and complete CIVER model. |
| `civic/ranking.py` | Intervention relevance and critique-subset selection. |
| `civic/objectives.py` | Seven joint training objectives. |
| `scripts/train_joint.py` | Joint training with separate module parameter groups. |
| `scripts/train_civic.py` | Original verifier–ranker experiment entry point. |
| `scripts/evaluate_matched_pairs.py` | Defective–repaired pair evaluation. |
| `scripts/evaluate_equivariance.py` | Transformation consistency evaluation. |
| `scripts/evaluate_finding_baselines.py` | Finding-validity baselines. |
| `scripts/evaluate_reviewer_cross_language.py` | Reviewer evaluation across languages. |
| `scripts/evaluate_ranking_baselines.py` | Intervention-retrieval baselines. |
| `scripts/evaluate_repair_controls.py` | Controlled downstream repair experiments. |

The `civic` import namespace and original training and evaluation workflows remain available. The original verifier–ranker trainer retains its historical behavior; use `train_joint` for the complete three-module objectives described above.

## Data and optional services

Benchmark datasets, intervention records, split manifests, frozen representations, and checkpoints are prepared separately. Existing experiments use workspace-relative paths under `data/`, `cache/`, `repos_cache/`, and `outputs/`; `CIVIC_PROJECT_ROOT` selects that workspace. Executable repair experiments also require their benchmark containers. Use disposable clones for checkout and patch experiments.

Optional service adapters read credentials, endpoints, model identifiers, and request settings from the local process environment. The empty template is in `env/.env.example`; scripts do not load it automatically. No service credentials or concrete model choices are supplied by the repository. Frozen-feature training runs without a service connection.

## Verification

```bash
python -m pytest -q
```

The offline checks cover the training graph, module gradients, state inputs, candidate masks, exact subset selection, and validation-based checkpointing. They verify the implementation and do not regenerate benchmark scores.
