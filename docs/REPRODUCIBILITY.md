# Reproducibility

## Required external artifacts

`CIVIC_PROJECT_ROOT` is a workspace containing `data/`, `cache/`, `outputs/`, and optional `repos_cache/`. Its default is the current working directory. Install the package with `pip install -e .` and run commands with `python -m scripts.<name>` or `python -m tools.<name>`. `CIVIC_PROJECT_ROOT` controls data and output locations independently of installed source code. The output directory is created automatically.

| Input | Used by |
| --- | --- |
| `data/intervention_dataset/quadruples_final.jsonl` | Intervention embedding and transformation scripts |
| `data/swe-review-bench_raw/data/*.parquet` | Patch lookup and executable experiments |
| `cache/embfix_index.jsonl` | Existing aligned index and split assignments required by the diff builder |
| `cache/embdiff_arrays.npz` | Arrays named `cx`, `cic`, `cicp`, `crit` |
| `cache/embdiff_index.jsonl` | Aligned metadata with `quad_id`, `issue_id`, `repo`, and `split` |
| `data/aacr-bench_raw/dataset.json` | AACR benchmark embedding builder |
| `cache/bench_aacr_emb.npz` | Finding arrays named `code`, `crit`, `labels` |
| `cache/bench_aacr_index.jsonl` | Aligned finding metadata and split assignments |
| `repos_cache/` | Dedicated disposable benchmark clones used by checkout/patch scripts |
| `data/anchors/` | Prior construction outputs used by historical repair/diagnostic scripts |

Quadruple records used by the diff builder include `quad_id`, `issue_id`, `critique` (text or an object containing `text`), and `I_cprime.from_issue`. Other scripts require additional intervention and localization fields. This table describes observed consumers, not a complete dataset release specification.

No download/preparation pipeline for all these artifacts is included. In particular, the diff builder inherits an existing `embfix_index.jsonl`; it does not create or validate the original project split. Check repository disjointness, duplicate issues, and split provenance before reporting generalization. Inspect exact field access in the selected script before preparing a dataset.

## Implementation boundaries

- Shared verifier definitions live in `civic/models.py`; the intervention ranker lives in `civic/ranking.py`.
- The main verifier combines projected code and critique vectors using a Transformer over two tokens. This is not a token-level code/critique attention model.
- In `train_civic.py`, the equivariance term is currently zero pending transformed inputs. Its minimality term is computed from frozen input differences and does not train the verifier. The full proposed seven-term objective is not implemented by that entry point.
- In `quad_eseries.py`, `clsonly` is a historical configuration name: no classification objective is optimized there. With no flip, specificity, or ranking loss, it serves as an untrained control because the remaining input-norm term has no gradient.
- The diff builder uses changed lines **and diff context**, rather than strictly changed lines alone. For unrelated edits outside the selected file, it reuses the original code representation. This can make specificity easier and must be considered when interpreting results.
- `e5_eqvset.py` is an exploratory unstable objective; `e5_eqvset_v2.py` is its later variant. Historical variants are retained and should not be treated as independent results.
- Some scripts perform expensive work at import time. Avoid importing arbitrary experiment modules for discovery.
- Benchmark-clone scripts reset and clean their working copies. Supply only dedicated disposable clones in `repos_cache/`.
- Dependency versions are minimums or unpinned. A validated lockfile, hardware manifest, released artifacts, and full end-to-end reproduction are not provided.

## Suggested reporting

Record benchmark versions, intervention acceptance criteria, project splits, random seeds, model/service identifiers, package versions, and compute. Separate finding-validity metrics from intervention-discrimination metrics. Report invalid or no-op intervention rates and scope failures alongside positive findings.
