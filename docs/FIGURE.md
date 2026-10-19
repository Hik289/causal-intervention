# Research figures

[Overview PNG](../assets/overview.png) · [Pipeline PDF](../assets/figures/joint_training_pipeline.pdf)

The overview illustrates the full CIVIC-CR research method. A frozen reviewer and intervention generator provide findings and candidate edits. Frozen encoders feed three jointly trained modules: a critique-conditioned verifier, an intervention ranker, and an evidence-aware review selector. The verifier supplies a shared state; ranking supervision updates that state; selection supervision flows through validity responses and soft intervention probabilities. The lower branch distinguishes target repairs, unrelated repairs, and mapped refactoring. The objective combines classification, flip, specificity, equivariance, ranking, minimality, and selection terms.

The illustration describes the research formulation. Individual scripts in this repository implement subsets; the illustration is not evidence that every module is available in a single released training entry point.

## Quantitative results

The six former bar charts have been replaced by readable tables, with the same reported numbers and green CIVIC-CR highlights. See the [results tables](RESULTS.md) for matched-pair discrimination, objective ablation, structured equivariance, downstream repair, real-PR filtering, and cross-language validity. Error ranges from the ablation and equivariance charts are preserved in the tables.

The remaining experimental figures show relationships across candidate-pool size or intervention coverage:

| Figure | Evaluation question |
| :--- | :--- |
| [Intervention retrieval](../assets/figures/intervention_retrieval.pdf) | Can the ranker discover the target among cross-example distractors? |
| [Intervention coverage](../assets/figures/intervention_coverage.pdf) | Which finding types admit meaningful target interventions? |

Intervention AUROC measures target-versus-unrelated score effects and differs from bug-versus-safe or finding-validity AUROC. The downstream 40× count ratio compares experimentally supplied precise guidance with false or absent guidance. Table exports do not replace the raw artifacts and split manifests required for reproduction.

## Illustration style and provenance

The overview was generated with the built-in image generation tool, using the supplied figurex examples as visual references. The brief called for a white background, hand-drawn outlines, restrained pastel panels, small code/document/robot symbols, legible module names, explicit forward connections, and backward learning paths. The final revision clarified the candidate-edit arrow into the intervention ranker and the shared response branches. No benchmark scores or conference logos were requested. The PDF preserves the same illustration for manuscript inclusion.
