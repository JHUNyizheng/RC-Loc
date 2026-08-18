# Release manifest

This folder is the complete GitHub upload set for the RC-Loc paper.

## Included

- Core Python package under `src/indoorloc`.
- Main training, baseline, shift, ablation, calibration, timing, and analysis scripts.
- Paper-matched and external-data configuration templates.
- Self-contained tests, dependency pins, documentation, citation metadata, and MIT license.
- Compact CSV/JSON results for clean accuracy, tail error, paired statistics, structured shifts, response surfaces, ablations, calibration, selective risk, and strict latency.

## Excluded by design

- Third-party raw datasets and derivative copies governed by their original licenses.
- Checkpoints, cached preprocessing arrays, training histories, and query-level predictions.
- Third-party source trees, virtual environments, temporary files, and compiled caches.
- Absolute workstation paths, credentials, tokens, and secret material.

The exclusions keep the repository auditable and within ordinary GitHub size limits. Users obtain each public corpus from its original distribution point and regenerate large intermediate artifacts with the documented commands.

