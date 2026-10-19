# Research figures

[Overview PNG](../assets/overview.png) · [Pipeline PDF](../assets/figures/joint_training_pipeline.pdf)

The overview illustrates the full CIVIC-CR research method. A frozen reviewer and intervention generator provide findings and candidate edits. Frozen encoders feed three jointly trained modules: a critique-conditioned verifier, an intervention ranker, and an evidence-aware review selector. The verifier supplies a shared state; ranking supervision updates that state; selection supervision flows through validity responses and soft intervention probabilities. The lower branch distinguishes target repairs, unrelated repairs, and mapped refactoring. The objective combines classification, flip, specificity, equivariance, ranking, minimality, and selection terms.

The illustration describes the research formulation. Individual scripts in this repository implement subsets; the illustration is not evidence that every module is available in a single released training entry point.

## Experimental plots

| Figure | Evaluation question |
| :--- | :--- |
| [Matched-pair discrimination](../assets/figures/matched_pair_discrimination.pdf) | Does one critique score the defective state above its repaired counterpart? |
| [Objective ablation](../assets/figures/objective_ablation.pdf) | Which losses support intervention discrimination and training stability? |
| [Structured equivariance](../assets/figures/structured_equivariance.pdf) | Do validity, location, and evidence remain consistent under mapped transformations? |
| [Downstream repair](../assets/figures/downstream_repair.pdf) | How valuable is accurate guidance to the same frozen repair agent? |
| [Intervention retrieval](../assets/figures/intervention_retrieval.pdf) | Can the ranker discover the target among cross-example distractors? |
| [Finding filtering](../assets/figures/finding_filtering.pdf) | Does verification reduce unsupported findings on real pull requests? |
| [Cross-language transfer](../assets/figures/cross_language_transfer.pdf) | Does a Python intervention verifier generalize to multilingual finding validity? |
| [Intervention coverage](../assets/figures/intervention_coverage.pdf) | Which finding types admit meaningful target interventions? |

The plots retain reported values, including negative and mixed outcomes. Intervention AUROC measures target-versus-unrelated score effects and differs from bug-versus-safe or finding-validity AUROC. The downstream 40× count ratio compares experimentally supplied precise guidance with false or absent guidance; it does not measure a 40× end-to-end learned-filter gain. Plot exports do not replace the raw artifacts and split manifests required for reproduction.

## Illustration style and provenance

The overview was generated with the built-in image generation tool, using the supplied figurex examples as visual references. The brief called for a white background, hand-drawn outlines, restrained pastel panels, small code/document/robot symbols, legible module names, explicit forward connections, and backward learning paths. The final revision clarified the candidate-edit arrow into the intervention ranker and the shared response branches. No benchmark scores or conference logos were requested. The PDF preserves the same illustration for manuscript inclusion.
