# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Merge Web of Science + Scopus, standardise fields, deduplicate.
"""
import pandas as pd
import numpy as np
import re
import unicodedata

# ---------- Config ----------
YEAR_MIN, YEAR_MAX = 2010, 2025
WOS_DT_KEEP = [
    'Article', 'Article; Data Paper', 'Article; Early Access',
    'Article; Proceedings Paper', 'Review', 'Proceedings Paper',
    'Review; Early Access',
]
SCOPUS_DT_KEEP = ['Article', 'Review', 'Conference Paper', 'Data Paper']

# ---------- Canonical schema (Scopus-style names) ----------
CANONICAL_COLS = [
    "Authors",                       # AU
    "Author full names",             # AF
    "Title",                         # TI
    "Source title",                  # SO
    "Abstract",                      # AB
    "Year",                          # PY
    "DOI",                           # DI
    "Cited by",                      # TC
    "Author Keywords",               # DE
    "Index Keywords",                # ID / Keywords Plus
    "Affiliations",                  # C1
    "Authors with affiliations",     # (Scopus) / built from C1 for WoS
    "Funding Details",               # FU
    "Correspondence Address",        # RP
    "Document Type",                 # DT
    "Abbreviated Source Title",      # JI / J9
    "Language",                      # LA
    # ---- derived / bookkeeping ----
    "Country",
    "Source_DB",
    "Source_Coverage",
    "Title_std",
    "DOI_std",
    "Has_DOI",
    "Decade",
]

# ---------- WoS -> canonical rename ----------
# ---------- WoS -> canonical rename ----------
WOS_RENAME = {
    'AU': 'Authors',
    'AF': 'Author full names',
    'TI': 'Title',
    'SO': 'Source title',
    'AB': 'Abstract',
    'PY': 'Year',
    'DI': 'DOI',
    'TC': 'Cited by',
    'DE': 'Author Keywords',
    'ID': 'Index Keywords',
    'C1': 'Affiliations',
    'RP': 'Correspondence Address',
    'FU': 'Funding Details',
    'DT': 'Document Type',
    'LA': 'Language',
    # 'J9' / 'JI' handled separately below to avoid duplicates
}

# ---------- Scopus -> canonical rename (mostly identity) ----------
SCOPUS_RENAME = {
    'Authors': 'Authors',
    'Author full names': 'Author full names',
    'Author(s) ID': 'Author IDs',
    'Title': 'Title',
    'Source title': 'Source title',
    'Abstract': 'Abstract',
    'Year': 'Year',
    'DOI': 'DOI',
    'Cited by': 'Cited by',
    'Author Keywords': 'Author Keywords',
    'Index Keywords': 'Index Keywords',
    'Affiliations': 'Affiliations',
    'Authors with affiliations': 'Authors with affiliations',
    'Funding Details': 'Funding Details',
    'Correspondence Address': 'Correspondence Address',
    'Document Type': 'Document Type',
    'Abbreviated Source Title': 'Abbreviated Source Title',
    'Language of Original Document': 'Language',
}




def _collapse_duplicate_columns(df):
    """If a rename produces two columns with the same name, merge them
    into one, keeping the first non-null value per row."""
    if not df.columns.duplicated().any():
        return df
    out = {}
    for col in df.columns.unique():
        same = df.loc[:, df.columns == col]
        if same.shape[1] == 1:
            out[col] = same.iloc[:, 0]
        else:
            # bfill across duplicate cols, then take the first column
            out[col] = same.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(out)

def rename_and_keep(df, rename_map):
    """Rename known columns to canonical names, collapse duplicates."""
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return _collapse_duplicate_columns(df)

def wos_add_abbrev_source(df):
    """Prefer JI (ISO abbrev) over J9; fold either into 'Abbreviated Source Title'."""
    if 'JI' in df.columns:
        df = df.rename(columns={'JI': 'Abbreviated Source Title'})
        if 'J9' in df.columns:
            df = df.drop(columns=['J9'])
    elif 'J9' in df.columns:
        df = df.rename(columns={'J9': 'Abbreviated Source Title'})
    return df



# =============================================================
# Helpers
# =============================================================
def log(stage, df, label):
    print(f"[{label}] {stage:<38s} -> {len(df):>7,d} rows")

def _norm_text(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = unicodedata.normalize('NFKC', s)   # keep accents but normalise
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else np.nan

def standardise_title_key(s):
    """Aggressive lowercase / de-accent / punctuation-free key for dedup."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else np.nan

def standardise_doi(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    s = re.sub(r'^https?://(dx\.)?doi\.org/', '', s)
    return s if s and s not in ('nan', 'none') else np.nan

def standardise_keywords(s):
    """Semicolon-separated, lowercased, trimmed, deduped, sorted."""
    if pd.isna(s):
        return np.nan
    s = str(s).replace(',', ';')  # some WoS exports use commas
    parts = [p.strip().lower() for p in s.split(';')]
    parts = [re.sub(r'\s+', ' ', p) for p in parts if p]
    parts = sorted(set(parts))
    return '; '.join(parts) if parts else np.nan

def standardise_abstract(s):
    """Trim, collapse whitespace, strip trailing copyright/©-notice."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = re.sub(r'\s+', ' ', s).strip()
    # remove trailing © 20xx ... publisher notices
    s = re.sub(r'\s*©.*$', '', s).strip()
    s = re.sub(r'\s*\bCopyright\b.*$', '', s, flags=re.I).strip()
    return s if s else np.nan

def standardise_authors(s):
    """Semicolon-separated author list; strip whitespace; unify separator."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    # WoS uses ';' between authors; Scopus uses ',' between authors
    # Heuristic: if ';' present, split on ';'; else split on ','
    sep = ';' if ';' in s else ','
    parts = [p.strip() for p in s.split(sep) if p.strip()]
    return '; '.join(parts) if parts else np.nan

def standardise_affiliations(s):
    """Collapse whitespace, standardise separator to '; '."""
    if pd.isna(s):
        return np.nan
    s = re.sub(r'\s+', ' ', str(s)).strip()
    # WoS wraps each affiliation in [Author] ... ; Scopus uses '; '
    parts = [p.strip() for p in re.split(r';\s*', s) if p.strip()]
    return '; '.join(parts) if parts else np.nan

# Country extraction from affiliation string ---------------------------------
_COUNTRY_FIXES = {
    'usa': 'United States', 'u.s.a.': 'United States', 'u.s.': 'United States',
    'united states of america': 'United States', 'united states': 'United States',
    'uk': 'United Kingdom', 'u.k.': 'United Kingdom',
    'england': 'United Kingdom', 'scotland': 'United Kingdom',
    'wales': 'United Kingdom', 'north ireland': 'United Kingdom',
    'peoples r china': 'China', "people's republic of china": 'China',
    'prc': 'China', 'russia': 'Russian Federation',
    'south korea': 'Republic of Korea', 'korea': 'Republic of Korea',
    'iran': 'Iran', 'viet nam': 'Vietnam',
}

def extract_country(aff):
    """Return semicolon-separated unique countries found in an affiliation string."""
    if pd.isna(aff):
        return np.nan
    countries = []
    for chunk in re.split(r';\s*', str(aff)):
        # last comma-separated token is usually the country
        parts = [p.strip() for p in chunk.split(',') if p.strip()]
        if not parts:
            continue
        cand = parts[-1]
        # strip US zip codes / postcodes
        cand = re.sub(r'\b\d{4,}\b.*$', '', cand).strip()
        cand = re.sub(r'\.$', '', cand).strip()
        key = cand.lower()
        cand = _COUNTRY_FIXES.get(key, cand.title())
        if cand:
            countries.append(cand)
    countries = sorted(set(countries))
    return '; '.join(countries) if countries else np.nan

def report_breakdown(df, source_name):
    print(f"\n===== {source_name}: breakdown =====")
    if 'Year' in df.columns:
        print("\nBy Year:");          print(df['Year'].value_counts().sort_index())
    if 'Document Type' in df.columns:
        print("\nBy Document Type:"); print(df['Document Type'].value_counts())
    if 'Language' in df.columns:
        print("\nBy Language:");      print(df['Language'].value_counts())

def rename_and_keep(df, rename_map):
    """Rename known columns to canonical names, keep everything else too."""
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# =============================================================
# 1. WEB OF SCIENCE
# =============================================================

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Merge and deduplicate Web of Science and Scopus exports.")
    parser.add_argument("--wos-csv", required=True, type=Path, help="Merged Web of Science CSV export.")
    parser.add_argument("--scopus-articles", required=True, type=Path, help="Scopus article/review CSV export (AR.csv).")
    parser.add_argument("--scopus-conference", required=True, type=Path, help="Scopus conference CSV export (Conf.csv).")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/MERGED_DEDUP_STANDARDISED.csv"), help="Output CSV path.")
    args = parser.parse_args()
    for input_path in (args.wos_csv, args.scopus_articles, args.scopus_conference):
        if not input_path.is_file():
            parser.error(f"Input file does not exist: {input_path}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    print("\n########## WEB OF SCIENCE ##########")
    wos = pd.read_csv(args.wos_csv, low_memory=False, encoding='utf-8-sig')
    log('raw load', wos, 'WOS')

    wos = rename_and_keep(wos, WOS_RENAME)
    for c in ('Document Type', 'Language'):
        if c in wos.columns:
            wos[c] = wos[c].astype(str).str.strip()

    wos_y   = wos[(wos['Year'] >= YEAR_MIN) & (wos['Year'] <= YEAR_MAX)]
    log(f'after year filter {YEAR_MIN}-{YEAR_MAX}', wos_y, 'WOS')

    wos_dt  = wos_y[wos_y['Document Type'].isin(WOS_DT_KEEP)]
    log('after document-type filter', wos_dt, 'WOS')

    wos_final = wos_dt[wos_dt['Language'] == 'English'].copy()
    log('after language filter (English)', wos_final, 'WOS')

    wos_final['Source_DB'] = 'WOS'
    report_breakdown(wos_final, 'WOS (filtered)')

    # =============================================================
    # 2. SCOPUS
    # =============================================================
    print("\n########## SCOPUS ##########")
    sc_ar   = pd.read_csv(args.scopus_articles, low_memory=False, encoding='utf-8-sig')
    sc_conf = pd.read_csv(args.scopus_conference, low_memory=False, encoding='utf-8-sig')
    scopus  = pd.concat([sc_ar, sc_conf], ignore_index=True)
    log('raw load (AR + Conf)', scopus, 'SCOPUS')

    scopus = rename_and_keep(scopus, SCOPUS_RENAME)
    for c in ('Document Type', 'Language'):
        if c in scopus.columns:
            scopus[c] = scopus[c].astype(str).str.strip()

    sc_y   = scopus[(scopus['Year'] >= YEAR_MIN) & (scopus['Year'] <= YEAR_MAX)]
    log(f'after year filter {YEAR_MIN}-{YEAR_MAX}', sc_y, 'SCOPUS')

    # (doc-type filter kept optional as in your script)
    sc_dt  = sc_y.copy()
    sc_final = sc_dt[sc_dt['Language'].str.contains('English', case=False, na=False)].copy() \
               if 'Language' in sc_dt.columns else sc_dt.copy()
    log('after language filter (English)', sc_final, 'SCOPUS')

    sc_final['Source_DB'] = 'SCOPUS'
    report_breakdown(sc_final, 'SCOPUS (filtered)')

    # =============================================================
    # 3. MERGE
    # =============================================================
    print("\n########## MERGE & STANDARDISATION ##########")
    n_wos, n_scopus = len(wos_final), len(sc_final)
    merged = pd.concat([wos_final, sc_final], ignore_index=True, sort=False)
    n_concat = len(merged)
    log('after concat (WOS + SCOPUS)', merged, 'MERGED')

    # =============================================================
    # 4. STANDARDISE FIELDS
    # =============================================================
    # Ensure every canonical column exists
    for c in CANONICAL_COLS:
        if c not in merged.columns:
            merged[c] = np.nan

    # --- Year: force int ---
    merged['Year'] = pd.to_numeric(merged['Year'], errors='coerce').astype('Int64')

    # --- Title & derived key ---
    merged['Title']     = merged['Title'].apply(_norm_text)
    merged['Title_std'] = merged['Title'].apply(standardise_title_key)

    # --- DOI ---
    merged['DOI_std']  = merged['DOI'].apply(standardise_doi)
    merged['DOI']      = merged['DOI_std']              # keep DOI clean
    merged['Has_DOI']  = merged['DOI'].notna()

    # --- Abstract ---
    merged['Abstract'] = merged['Abstract'].apply(standardise_abstract)

    # --- Keywords ---
    merged['Author Keywords'] = merged['Author Keywords'].apply(standardise_keywords)
    merged['Index Keywords']  = merged['Index Keywords'].apply(standardise_keywords)

    # --- Authors ---
    merged['Authors']            = merged['Authors'].apply(standardise_authors)
    merged['Author full names']  = merged['Author full names'].apply(standardise_authors)

    # --- Affiliations ---
    merged['Affiliations']              = merged['Affiliations'].apply(standardise_affiliations)
    merged['Authors with affiliations'] = merged['Authors with affiliations'].apply(standardise_affiliations)

    # --- Country (derived from Affiliations) ---
    merged['Country'] = merged['Affiliations'].apply(extract_country)

    # --- Source title / abbrev / other free-text ---
    for c in ('Source title', 'Abbreviated Source Title',
              'Correspondence Address', 'Funding Details',
              'Document Type', 'Language'):
        merged[c] = merged[c].apply(_norm_text)

    # --- Cited by: numeric ---
    merged['Cited by'] = pd.to_numeric(merged['Cited by'], errors='coerce').fillna(0).astype(int)

    # --- Decade ---
    merged['Decade'] = (merged['Year'] // 10 * 10).astype('Int64')

    # =============================================================
    # 5. DEDUPLICATE (DOI first, then title key)
    # =============================================================
    with_doi    = merged.dropna(subset=['DOI']).drop_duplicates(subset=['DOI'])
    without_doi = merged[merged['DOI'].isna()].drop_duplicates(subset=['Title_std'])
    dedup = pd.concat([with_doi, without_doi], ignore_index=True)
    dedup = dedup.drop_duplicates(subset=['Title_std'])

    n_final      = len(dedup)
    n_duplicates = n_concat - n_final
    pct_removed  = n_duplicates / n_concat * 100 if n_concat else 0

    # Source coverage tag ---------------------------------------------------------
    overlap = (merged.groupby('Title_std')['Source_DB']
                     .agg(lambda x: '+'.join(sorted(set(x))))
                     .rename('Source_Coverage'))
    dedup = dedup.drop(columns=['Source_Coverage'], errors='ignore') \
                 .merge(overlap, on='Title_std', how='left')

    n_both        = (dedup['Source_Coverage'] == 'SCOPUS+WOS').sum()
    n_wos_only    = (dedup['Source_Coverage'] == 'WOS').sum()
    n_scopus_only = (dedup['Source_Coverage'] == 'SCOPUS').sum()

    # =============================================================
    # 6. SELECT FINAL COLUMNS
    # =============================================================
    final_cols = [c for c in CANONICAL_COLS if c in dedup.columns]
    dedup = dedup[final_cols].copy()

    # =============================================================
    # 7. REPORT
    # =============================================================
    print("\n" + "=" * 60)
    print(" DEDUPLICATION SUMMARY (quotable)")
    print("=" * 60)
    print(f" Web of Science (after filters) : {n_wos:>7,d}")
    print(f" Scopus         (after filters) : {n_scopus:>7,d}")
    print(f" Combined (before deduplication): {n_concat:>7,d}")
    print(f" Duplicates removed             : {n_duplicates:>7,d}  ({pct_removed:5.2f}%)")
    print(f" UNIQUE DOCUMENTS AFTER DEDUP   : {n_final:>7,d}")
    print("-" * 60)
    print(f"   Indexed in BOTH databases    : {n_both:>7,d}")
    print(f"   WOS only                     : {n_wos_only:>7,d}")
    print(f"   Scopus only                  : {n_scopus_only:>7,d}")
    print("=" * 60)

    print(
        f"\nAfter merging Web of Science (n={n_wos:,}) and Scopus (n={n_scopus:,}) "
        f"records and removing {n_duplicates:,} duplicates ({pct_removed:.2f}%), "
        f"a total of {n_final:,} unique documents were retained for analysis "
        f"({n_both:,} indexed in both databases, {n_wos_only:,} in WoS only, "
        f"{n_scopus_only:,} in Scopus only)."
    )

    print("\nFinal by year:");          print(dedup['Year'].value_counts().sort_index())
    print("\nFinal by document type:"); print(dedup['Document Type'].value_counts())
    print("\nTop 15 countries:")
    country_series = (dedup['Country'].dropna()
                                       .str.split('; ')
                                       .explode()
                                       .value_counts()
                                       .head(15))
    print(country_series)

    # =============================================================
    # 8. SAVE
    # =============================================================
    out = args.output_csv
    dedup.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\nSaved -> {out}   ({n_final:,} rows, {dedup.shape[1]} cols)")
    print("Columns kept:", list(dedup.columns))

if __name__ == "__main__":
    main()
