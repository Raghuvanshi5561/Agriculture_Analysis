# Agriculture 4.0 bibliometric analysis pipeline

Reproducible Python workflow for merging Web of Science (WoS) and Scopus exports, filtering and deduplicating records, and generating keyword co-occurrence networks and time-windowed thematic maps.

## Repository structure

- `run_pipeline.py` — single entry point for the full workflow.
- `src/merge_wos_scopus.py` — canonical field harmonisation, merge, and deduplication.
- `src/filter_corpus.py` — year, document-type, language, and optional publication-stage filtering.
- `src/cooccurrence_network.py` — keyword co-occurrence network analysis and exports.
- `src/thematic_mapping.py` — time-windowed thematic mapping and exports.
- `src/RCA_code.py` — **RCA only**: country-by-technology RCA matrices, binary specialisation heatmaps, tables, and LaTeX output.
- `src/rca_country_ranking_sensitivity.py` — RCA-based country ranking and sensitivity analysis (RCA cut-offs, publication thresholds, two-dimensional grids, and ranking robustness).
- `src/country_technology_share.py` — country-level technology-domain composition based on the three-layer rule-based classified CSV; exports a 100% stacked PDF and country-level count/share CSV.
- `src/domain_share_over_time.py` — annual technology-domain shares using fractional counting; exports a 100% stacked area chart (PDF/PNG) and annual share table.
- `data/` — place local input files here if you are authorised to use them; data are not tracked by Git.
- `outputs/` — generated files; outputs are not tracked by Git.

## 1. Install

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## 2. Prepare input exports

Export bibliographic records from WoS and Scopus to CSV. The workflow expects:

- one merged WoS CSV (e.g. `WOS_MERGED.csv`),
- one Scopus article/review CSV (e.g. `AR.csv`),
- one Scopus conference CSV (e.g. `Conf.csv`).

Do not upload raw WoS/Scopus exports to a public repository unless your licence explicitly permits redistribution. Keep them outside Git or in the ignored `data/` directory.

## 3. Run the full workflow with one command

```bash
python run_pipeline.py --wos-csv data/WOS_MERGED.csv --scopus-articles data/AR.csv --scopus-conference data/Conf.csv
```

By default, the pipeline writes the merged corpus, filtered corpus, filter log, co-occurrence outputs, and thematic-mapping outputs to `outputs/`. The examples below use repository-relative paths so the project can be run on Windows, macOS, or Linux without editing the source code. If your input files are stored elsewhere, pass their paths as command-line arguments; no machine-specific directories are hard-coded in the scripts.

```bash
python run_pipeline.py --wos-csv data/WOS_MERGED.csv --scopus-articles data/AR.csv --scopus-conference data/Conf.csv --output-dir outputs/results
```

To run only the merge/filter stages, use `--skip-cooccurrence --skip-thematic`. To skip just one analysis, use `--skip-cooccurrence` or `--skip-thematic`. Use `--keep-in-press` to retain in-press records when the input contains a `Publication Stage` column; otherwise the filter keeps final-stage records when that column exists.

## 4. Run individual stages

```bash
python src/merge_wos_scopus.py --wos-csv data/WOS_MERGED.csv --scopus-articles data/AR.csv --scopus-conference data/Conf.csv --output-csv outputs/MERGED_DEDUP_STANDARDISED.csv
python src/filter_corpus.py --input-csv outputs/MERGED_DEDUP_STANDARDISED.csv --output-csv outputs/MERGED_FILTERED.csv --log-csv outputs/filter_log.csv
python src/cooccurrence_network.py --input-csv outputs/MERGED_FILTERED.csv --output-dir outputs/cooccurrence
python src/thematic_mapping.py --input-csv outputs/MERGED_FILTERED.csv --output-dir outputs/thematic_mapping
```


## 5. RCA and country-ranking analyses (separate workflows)

The repository contains two distinct RCA-related scripts. They are intentionally separate because they answer different analytical questions:

- **RCA-only analysis** (`src/RCA_code.py`): calculates and visualises revealed comparative advantage across technology domains. It does not run the country-ranking sensitivity workflow.
- **Country ranking and sensitivity** (`src/rca_country_ranking_sensitivity.py`): calculates RCA-based country rankings and tests their stability under alternative RCA cut-offs, publication thresholds, and ranking specifications.

Both scripts use the classified publication-level CSV and expect these columns by default: `Year`, `countries`, and `tech_domains`. Country and domain values are expected to be semicolon-separated where multiple values occur in a cell. The ranking script uses the seven canonical domain labels configured in its source; labels in the input must match those labels exactly.

Run the RCA-only workflow:

```bash
python src/RCA_code.py --input-csv data/three_layer_classified_comprehensive_normalized.csv --output-dir outputs/rca_only
```

Run the country-ranking sensitivity workflow:

```bash
python src/rca_country_ranking_sensitivity.py --input-csv data/three_layer_classified_comprehensive_normalized.csv --output-dir outputs/rca_sensitivity
```

The output directory can be changed to any writable path. Do not commit classified publication data unless you have the right to redistribute it. Review the configured time windows, domain labels, baseline cut-offs, publication thresholds, and minimum-output rules before interpreting or reporting results.

## 6. Country-level technology composition and annual domain shares

These scripts use the three-layer rule-based classification output. They are separate from the RCA-only and RCA-based country-ranking sensitivity analyses because they describe technology composition/shares rather than revealed comparative advantage.

### Country-by-technology composition

`src/country_technology_share.py` expects `countries`, `tech_domains`, and optionally `DOI`. Country lists may be semicolon- or comma-separated; technology labels must be semicolon-separated and begin with domain codes `D1`–`D7`. It counts each distinct publication-country pair once and each publication-country-domain assignment once, then calculates the domain share within each country's assignments.

```bash
python src/country_technology_share.py --input-csv data/three_layer_classified_comprehensive_normalized.csv --output-dir outputs/country_technology_share --top-n 20
```

Outputs include a PDF stacked-bar figure and a CSV with country publication counts, domain counts, and within-country shares.

### Annual technology-domain share

`src/domain_share_over_time.py` expects `Year`, `tech_n`, and the seven Boolean/numeric columns `TECH_D1_AI_and_machine_learning` through `TECH_D7_Blockchain`. Each publication's domain indicators are divided by `tech_n` (fractional counting), aggregated by year, and normalised to 100% within each year.

```bash
python src/domain_share_over_time.py --input-csv data/three_layer_classified_comprehensive_normalized.csv --output-dir outputs/domain_share --year-min 2010 --year-max 2025
```

Outputs include PDF and PNG figures and an annual percentage-share CSV. The scripts require the classified dataset to have the expected column names; inspect the column names and the classification rules before running them on another dataset.

## 7. Three-layer rule-based classification

`src/three_layer_classification.py` applies the project's dictionary-based, multi-label classification to publication metadata. It includes built-in self-tests and writes five outputs: the full classified corpus, the P3 policy-priority subset, a field-level classification audit, unclassified records, and a category summary.

Run it on a CSV containing a title column (required). Abstract, author-keyword, indexed-keyword, year, and country columns are detected from common column-name variants when available:

```bash
python src/three_layer_classification.py data/three_layer_classification_input.csv --output-dir outputs/three_layer_classification
```

Classification rules implemented in the script:

- **Technology domains (D1–D7):** title, author keywords, and indexed keywords are checked first. Abstract text is used only as a fallback when none of those fields identifies any technology domain.
- **Policy priorities (P1–P5):** title, author keywords, indexed keywords, and abstract are checked.
- **Environmental outcome categories (E1–E5):** title, author keywords, indexed keywords, and abstract are checked.
- A category is assigned when its regular-expression dictionary matches in a permitted field. There is no weighted score or category threshold. A publication can receive multiple labels.
- Field-specific evidence columns are retained to support auditing. A text match indicates that the topic is represented in the record; it does not by itself establish a measured or demonstrated real-world outcome.

The input CSV is not included. Use your authorised source dataset and inspect the script's dictionaries and the resulting audit before relying on classifications in a publication. The classifier's self-tests check selected positive/negative examples and field logic; they are not a substitute for independent, category-level validation against manually coded records.

## Important methodological notes

- The default publication-year range is 2010–2025. Exact document-type filtering retains `Article`, `Review`, `Conference paper`, and `Proceedings Paper`; compound document-type labels are excluded by the filter.
- The merger has its own source-specific document-type and English-language filters. Review these settings against the methods reported in the manuscript before reproducing results.
- The filter accepts either `Language` (the canonical field produced by the merger) or `Language of Original Document` (the original Scopus field). It applies an exact English-language match and drops blank language values.
- Deduplication uses DOI first and normalized title as a secondary key. As with any bibliographic deduplication, inspect edge cases and verify counts before reporting them.
- Network thresholds, keyword normalization dictionaries, and time windows are configured in the analysis scripts. Preserve the code version used for manuscript figures.
- The pipeline was syntax-checked but not end-to-end validated against the licensed source exports. Results may vary with input exports and package versions.

## Licence and data availability

Before making the repository public, choose a licence for your code and verify that you have the right to license all included material. A code licence does not override WoS or Scopus database terms. Do not commit credentials, private data, or licensed raw exports. Source files use command-line input/output paths and repository-relative examples; do not add personal computer paths (for example, a user home folder or Desktop directory) to committed code.
