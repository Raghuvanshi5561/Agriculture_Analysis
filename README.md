# Agriculture 4.0 bibliometric analysis workflow

Repository containing the scripts for corpus preparation, three-layer classification, bibliometric mapping, technology-domain shares, and country-level RCA analysis.

## Repository structure

```text
.
├── .github/workflows/python-ci.yml
├── data/                  # Place local input exports/CSVs here; data files are git-ignored
├── outputs/               # Generated outputs; ignored by Git
├── scripts/
│   ├── merge_wos_scopus.py
│   ├── filter_corpus.py
│   ├── three_layer_classification.py
│   ├── BIB_NEW_PLOTS_frac_updated.py
│   ├── fig_domain_env_custom_updated.py
│   ├── fig_domain_policy_updated.py
│   ├── fig_sankey_tech_env_policy.py
│   ├── cooccurrence_network.py
│   ├── thematic_mapping.py
│   ├── country_technology_share.py
│   ├── domain_share_over_time.py
│   ├── rca_analysis.py
│   └── rca_country_ranking_sensitivity.py
├── requirements.txt
└── README.md
```

## Setup

Python 3.10–3.12 is recommended. Create an environment and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## End-to-end workflow

The pipeline is staged. Input data are not included in this repository; export/download them from the licensed sources you use and put them in `data/`. Do not commit restricted bibliographic records.

### Stage 1 — Merge Web of Science and Scopus exports

Supply the three exports explicitly. File names below are examples; adjust them to match the files in your local `data/` folder.

```bash
python scripts/merge_wos_scopus.py \
  --wos-csv data/wos_export.csv \
  --scopus-articles data/scopus_articles.csv \
  --scopus-conference data/scopus_conference.csv \
  --output-csv outputs/MERGED_DEDUP_STANDARDISED.csv
```

### Stage 2 — Filter the merged corpus

```bash
python scripts/filter_corpus.py \
  --input-csv outputs/MERGED_DEDUP_STANDARDISED.csv \
  --output-csv outputs/MERGED_FILTERED.csv \
  --log-csv outputs/filter_log.csv \
  --year-min 2010 --year-max 2025 --language English
```

### Stage 3 — Classify technology, policy, production, and environmental dimensions

```bash
python scripts/three_layer_classification.py \
  outputs/MERGED_FILTERED.csv \
  --output-dir outputs/classification
```

This script writes its classified corpus and audit/summary outputs to `outputs/classification/`. For the analysis scripts below, use the generated classified CSV that matches your chosen corpus and column schema. If your existing file is named `three_layer_classified_FINAL_v4.csv`, place it in `data/` or pass it as `--input-csv` to scripts that accept that option.

### Stage 4 — Main manuscript figures and cross-tabulations

The four scripts below use `data/three_layer_classified_FINAL_v4.csv` by default and write outputs below `outputs/`:

```bash
python scripts/BIB_NEW_PLOTS_frac_updated.py
python scripts/fig_domain_env_custom_updated.py
python scripts/fig_domain_policy_updated.py
python scripts/fig_sankey_tech_env_policy.py
```

### Stage 5 — Thematic mapping and co-occurrence networks

```bash
python scripts/thematic_mapping.py \
  --input-csv data/three_layer_classified_FINAL_v4.csv \
  --output-dir outputs/thematic_mapping

python scripts/cooccurrence_network.py \
  --input-csv data/three_layer_classified_FINAL_v4.csv \
  --output-dir outputs/cooccurrence
```

Both scripts use NLTK English stopwords; run the downloader command in Setup before execution.

### Stage 6 — Technology-domain composition and annual shares

`country_technology_share.py` uses `countries` and `tech_domains`:

```bash
python scripts/country_technology_share.py \
  --input-csv data/three_layer_classified_FINAL_v4.csv \
  --output-dir outputs/country_technology_share
```

`domain_share_over_time.py` expects the binary `TECH_D1`–`TECH_D7` indicator columns and `tech_n`. It therefore requires the corresponding normalized/comprehensive classification export, not necessarily the compact three-layer file:

```bash
python scripts/domain_share_over_time.py \
  --input-csv data/three_layer_classified_comprehensive_normalized.csv \
  --output-dir outputs/domain_share \
  --year-min 2010 --year-max 2025
```

### Stage 7 — RCA and country-ranking robustness

```bash
python scripts/rca_analysis.py \
  --input-csv data/three_layer_classified_FINAL_v4.csv \
  --output-dir outputs/rca

python scripts/rca_country_ranking_sensitivity.py \
  --input-csv data/three_layer_classified_FINAL_v4.csv \
  --output-dir outputs/rca_sensitivity
```

The RCA scripts expect the `countries` and `tech_domains` fields. Review each script's command-line help (`--help`) for optional settings.

## GitHub Actions workflow

`.github/workflows/python-ci.yml` installs the dependencies, downloads NLTK stopwords, compiles every Python script, and checks that no personal workstation paths remain. It does not run the analyses because the source datasets are intentionally excluded from Git.

## Data, reproducibility, and licensing

- All input and output paths should be supplied as command-line arguments or resolved relative to the repository; no personal workstation paths are required.
- The `data/` and `outputs/` contents are ignored by Git by default. This avoids accidental publication of licensed bibliographic exports or large derived files.
- Keep the exact data extraction dates, database versions, query strings, and classification settings in a separate reproducibility note if needed for the manuscript.
- Syntax and CI checks do not validate scientific results. Run the scripts against your full data and inspect the generated audit tables, figures, and warnings before publication.
