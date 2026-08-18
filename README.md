# RC-Loc
Leveraging Mask Perception and Corruption Consistency for Wi-Fi Fingerprint Localization With Indoor Radio Maps

Official reproducibility package for **“RC-Loc: Leveraging Mask Perception and Corruption Consistency for Wi-Fi Fingerprint Localization With Indoor Radio Maps”**

RC-Loc separates received signal strength (RSS) from access-point (AP) availability, trains on paired clean and corrupted fingerprints, and combines a direct coordinate estimate with validation-selected retrieval over surveyed radio-map anchors. This folder contains the source code, tests, experiment entry points, configurations, fixed seed protocols, and compact CSV/JSON records used in the paper.

## Release contents

- `src/indoorloc/`: data loaders, data adapters, models, metrics, and training utilities.
- `scripts/`: clean-accuracy, baseline, structured-shift, ablation, calibration, latency, and analysis programs.
- `configs/`: the paper training configuration and an external long-format fingerprint template.
- `tests/`: self-contained unit tests for data processing, metrics, adapters, and models.
- `results_summary/`: GitHub-sized numerical data and run manifests used to trace the reported results.
- `docs/`: the experimental protocol and evidence limitations.
- `data/README.md`: required raw-data layout and redistribution boundary.

The package intentionally excludes third-party raw datasets, trained checkpoints, preprocessing caches, training histories, and query-level prediction arrays. Raw datasets remain subject to their providers’ licenses, while checkpoints and predictions are too large for a compact source repository. The included aggregate data are sufficient to audit the reported means, tail metrics, paired statistics, response surfaces, ablations, calibration, selective risk, and latency measurements.

## Experimental scope

- Datasets: UJIIndoorLoc, UTSIndoorLoc, Tampere, and UJI-Library-25M.
- Main training seeds: `11 22 33 44 55`.
- Independent calibration seeds: `11 22 33 44 55 66 77 88 99 111`.
- RC-Loc training: 200-epoch ceiling, 50-epoch minimum, patience 30, batch size 256.
- Model selection: an 18% spatial-group validation split constructed independently for each seed.
- Evaluation: clean localization, correlated AP outage, persistent AP bias, global RSS offset, joint shifts, component ablations, 24-month forward time, calibration, selective risk, and strict end-to-end latency.

## Environment

The reported experiments used Python 3.11 and PyTorch 2.5.0. Install the numerical dependencies first, then choose the PyTorch wheel appropriate for the local CPU/GPU platform.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Example CUDA 12.1 build; select a different official wheel when appropriate.
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121

$env:PYTHONPATH = "$PWD\src"
pytest -q
```

The test suite does not require the public datasets. PyTorch may emit nested-tensor optimization warnings for the transformer baseline; these warnings do not change the evaluated outputs.

## Raw-data preparation

No public-dataset files are redistributed in this repository. Obtain each dataset from its provider, review its license, and place the extracted files below the repository root.

| Dataset | Provider/download page |
|---|---|
| UJIIndoorLoc | https://archive.ics.uci.edu/dataset/310/ujiindoorloc |
| UTSIndoorLoc | https://github.com/XudongSong/UTSIndoorLoc-dataset/tree/master/UTSIndoorLoc |
| Tampere WLAN RSS | https://doi.org/10.5281/zenodo.1161525 |
| UJI-Library-25M (`UJI_LIB_DB_v2.2`) | https://doi.org/10.5281/zenodo.3748719 |

After downloading, use this layout:

```text
data/raw/
├── UJIIndoorLoc/UJIndoorLoc/{trainingData.csv,validationData.csv}
├── UTSIndoorLoc/UTSIndoorLoc/{UTS_training.csv,UTS_test.csv}
├── Tampere/{Training_rss.csv,Test_rss.csv,Training_coordinates.csv,Test_coordinates.csv}
└── UJI_Library/extracted/db/{01,...,25}/{*rss.csv,*crd.csv}
```

`src/indoorloc/twc_data.py` implements the exact parsing and normalization rules. UJI-Library month 1 is used for fitting, and months 2–25 are evaluated in chronological order without target-month updating.

## Main reproduction commands

Run commands from the repository root. Output is written below `results/`. Completed dataset/seed pairs are skipped when their recorded output is present.

```powershell
$env:PYTHONPATH = "$PWD\src"

# RC-Loc: four datasets × five complete training seeds
python scripts/run_rc_final.py --datasets uji uts tampere uji_library `
  --seeds 11 22 33 44 55 --max-epochs 200 --patience 30

# Protocol-matched controls
python scripts/replay_classical_predictions.py
python scripts/run_anchor_transformer_baseline.py
python scripts/run_modern_rss_baselines.py

# Structured shifts, component studies, calibration, and service cost
python scripts/run_ood_robustness.py
python scripts/run_ood_response_surface.py
python scripts/run_rc_component_ablations.py
python scripts/run_independent_calibration.py
python scripts/benchmark_strict_end_to_end.py

# Rebuild compact numerical summaries and figure inputs from saved predictions
python scripts/analyze_twc_extended.py
```

The full run is computationally intensive. Do not reduce epochs or seeds when reproducing the paper’s reported statistics. See `docs/experiment_protocol.md` and the JSON manifests under `results_summary/` for the exact evaluation definitions.

## Result traceability

`results_summary/twc_revision/` records the five-seed clean and shift results, paired statistics, tuning choices, and ten independent calibration splits. `results_summary/twc_extended/` records the additional baselines, multi-metric ranks, joint outage–bias surface, component ablations, calibration reliability, selective risk, and strict latency environment.

Every result row identifies its dataset, seed, method, scenario, sample count, and reported metrics. Manifest files record the generation program and protocol boundaries. Query-level predictions are excluded from GitHub; regenerating them requires the public datasets and the commands above.

## External fingerprint data

`configs/huawei_template.yaml` documents the expected long-format schema for a separate deployment dataset. It is a template only and contains no proprietary measurements. The adapter in `src/indoorloc/adapters.py` converts the long table into the aligned AP matrix used by the model.

## Citation

If you use this package, cite the accompanying manuscript and the tagged software release. Machine-readable citation metadata are provided in `CITATION.cff`.

```text
Yi Zheng, “RC-Loc: Mask-Aware Corruption-Consistent Wi-Fi Fingerprint
Localization under Anchor-Set and RSS Distribution Shifts,” submitted to
IEEE Internet of Things Journal, 2026.
```

## License

The source code is released under the MIT License. Dataset licenses, terms, and attribution requirements remain with the original dataset providers. The numerical summaries are research artifacts associated with the manuscript and should be cited with the software release.
