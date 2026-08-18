# Dataset links and placement

Raw fingerprints are not redistributed in this repository. Use the provider pages below and review each dataset license before downloading.

| Dataset | Provider/download page |
|---|---|
| UJIIndoorLoc | https://archive.ics.uci.edu/dataset/310/ujiindoorloc |
| UTSIndoorLoc | https://github.com/XudongSong/UTSIndoorLoc-dataset/tree/master/UTSIndoorLoc |
| Tampere WLAN RSS | https://doi.org/10.5281/zenodo.1161525 |
| UJI-Library-25M (`UJI_LIB_DB_v2.2`) | https://doi.org/10.5281/zenodo.3748719 |

Place the extracted files under `data/raw/` using the structure below.

```text
data/raw/
├── UJIIndoorLoc/UJIndoorLoc/{trainingData.csv,validationData.csv}
├── UTSIndoorLoc/UTSIndoorLoc/{UTS_training.csv,UTS_test.csv}
├── Tampere/{Training_rss.csv,Test_rss.csv,Training_coordinates.csv,Test_coordinates.csv}
└── UJI_Library/extracted/db/{01,...,25}/{*rss.csv,*crd.csv}
```

The loader implementation in `src/indoorloc/twc_data.py` is authoritative for file names, coordinate fields, missing-value handling, normalization, and chronological partitioning. Do not commit the contents of `data/raw/`; the repository `.gitignore` excludes that directory.

Compact numerical data used by the paper are already included in `results_summary/`.
