# RC-Loc experimental protocol

## Data and partitioning

- UJIIndoorLoc: 19,937 training fingerprints, 1,111 official test fingerprints, and 520 APs.
- UTSIndoorLoc: 9,108 training fingerprints, 388 official route-test fingerprints, and 589 APs.
- Tampere: 1,478 training fingerprints, 489 official trajectory-test fingerprints, and 312 APs after parsing the released matrices.
- UJI-Library-25M: month 1 is used for fitting; months 2–25 are evaluated chronologically without target-month updating.

The main training seeds are 11, 22, 33, 44, and 55. For every seed, `GroupShuffleSplit` holds out 18% of the training spatial groups for validation. Fingerprints sharing a coordinate, floor, and building remain on the same side of the fit/validation partition. Scaling, PCA, early stopping, AP ranking, graph selection, fusion selection, and calibration do not use official test labels.

## Model and training budget

RC-Loc uses separate RSS and observation-mask stems, a 96-dimensional embedding, paired clean/corrupted coordinate supervision, prediction and embedding consistency, observed-RSS reconstruction, and coarse floor/building heads. A spatially validated blend combines the direct coordinate head with exact retrieval over fit-set radio-map anchors.

AdamW uses learning rate `7e-4`, weight decay `2e-4`, batch size 256, cosine decay, and gradient-norm clipping at 5. The supervised run has a 200-epoch ceiling, a 50-epoch minimum, and patience 30. Early stopping uses the spatial-validation objective `median error + 0.2 × P90`.

## Validation matrix

- Official clean test partitions on four datasets with five complete training seeds.
- UJI/UTS structured loss of frequently observed fit-set APs.
- Persistent AP-specific RSS bias, device-wide RSS offsets, and outage–bias compositions.
- RC-Loc component removals under clean and shifted conditions.
- UJI-Library months 2–25 evaluated in chronological order without target-domain updates.
- Ten independent calibration splits for 90% empirical-radius coverage.
- Neighborhood risk, selective risk, parameter count, training time, and warmed end-to-end batch latency.

The primary endpoints are two-dimensional Euclidean mean error and P90. Median, P95, P99, threshold success, floor/building accuracy, calibration coverage, and CVaR are reported where applicable. Statistical inference treats the five seeded training/model-selection repetitions as the independent units. The retained summaries include paired wins, the exact directional sign-flip test over all `2^5` assignments, and 50,000-resample seed-level bootstrap intervals.

## Reproduction records

The complete training scripts write models, histories, split metadata, tuning decisions, predictions, and metrics beneath `results/`. GitHub carries compact CSV/JSON summaries and manifests in `results_summary/`; model checkpoints and query-level predictions can be regenerated from the public datasets.

