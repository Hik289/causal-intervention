# Release validation

The initial public release was prepared with the following checks:

- Parsed all 65 Python files and verified that Python comment tokens and standalone documentation strings were removed. Assigned prompt strings and functional text were preserved.
- Compared syntax trees against the original sources after accounting for the documented environment, path, and output-format changes. The remaining program structure matched.
- Scanned release text for personal absolute paths, named private identifiers, email addresses, private-key headers, and common credential patterns. No matching sensitive material remained in the scanned release text. Pattern scanning cannot establish the absence of every possible secret.
- Replaced placeholder API credentials and local proxy defaults with explicit process environment variables. No private service address is required by the published examples.
- Corrected phase-directory references to match this repository layout, added missing `anthropic` and Parquet support dependencies, and prevented a missing specificity-ratio value from crashing ablation-result formatting.
- Checked finite forward outputs and gradients for MLP, bilinear, and Transformer verifier architectures using synthetic inputs.
- Successfully invoked the main training command's help interface in the available PyTorch environment.
- Excluded data, caches, model weights, generated results, local environment files, and operating-system metadata from version control.

Full benchmark training, intervention construction, external service calls, and Docker repair experiments were not run: required benchmark artifacts and credentials were not supplied. These checks validate source cleanup and basic model execution, not paper results or freedom from methodological leakage. Dataset provenance and train/test independence require separate validation with the actual artifacts.
