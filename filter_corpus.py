#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter the merged bibliographic corpus by year, document type, and language."""
import argparse
from pathlib import Path
import pandas as pd

KEEP_TYPES_EXACT = {"article", "review", "conference paper", "proceedings paper"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path, help="Merged bibliographic CSV.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/MERGED_FILTERED.csv"))
    parser.add_argument("--log-csv", type=Path, default=Path("outputs/filter_log.csv"))
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2025)
    parser.add_argument("--language", default="English")
    parser.add_argument("--keep-in-press", action="store_true", help="Retain records marked as in press when Publication Stage exists.")
    parser.add_argument("--keep-references", action="store_true", help="Retain the potentially large References column.")
    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f"Input file does not exist: {args.input_csv}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(args.input_csv, nrows=0, encoding="utf-8-sig")
    usecols = [c for c in header.columns if args.keep_references or c != "References"]
    df = pd.read_csv(args.input_csv, usecols=usecols, low_memory=False, encoding="utf-8-sig")
    logs=[]
    def step(name, frame):
        logs.append({"step": name, "records": int(len(frame))})
        print(f"{name:<58} {len(frame):>8,d}")
    print("FILTERING MERGED CORPUS")
    step("Records loaded", df)
    if "Year" not in df.columns:
        parser.error("Required column 'Year' is missing.")
    years = pd.to_numeric(df["Year"], errors="coerce")
    missing_years = int(years.isna().sum())
    df = df.loc[years.between(args.year_min, args.year_max)].copy()
    step(f"Year {args.year_min}-{args.year_max} (missing/non-numeric years: {missing_years})", df)
    if "Document Type" not in df.columns:
        parser.error("Required column 'Document Type' is missing.")
    dt = df["Document Type"].fillna("").astype(str).str.lower().str.strip()
    excluded = df.loc[~dt.isin(KEEP_TYPES_EXACT), "Document Type"].value_counts(dropna=False)
    df = df.loc[dt.isin(KEEP_TYPES_EXACT)].copy()
    step("Exact document-type filter", df)
    if len(excluded):
        print("Excluded document types:")
        for label, count in excluded.items():
            print(f"  {label}: {count}")
    # Merge stage standardizes the field to Language; accept original Scopus name too.
    language_col = next((c for c in ("Language", "Language of Original Document") if c in df.columns), None)
    if language_col is None:
        parser.error("No language column found. Expected 'Language' or 'Language of Original Document'.")
    language = df[language_col].fillna("").astype(str).str.strip()
    missing_language = int(language.eq("").sum())
    df = df.loc[language.str.casefold().eq(args.language.casefold())].copy()
    step(f"Language filter: {args.language} (blank before filter: {missing_language})", df)
    if "Publication Stage" in df.columns and not args.keep_in_press:
        stage = df["Publication Stage"].fillna("").astype(str).str.casefold().str.strip()
        df = df.loc[stage.eq("final")].copy()
        step("Publication stage: Final", df)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(logs).to_csv(args.log_csv, index=False, encoding="utf-8-sig")
    print(f"\nFinal records: {len(df):,}")
    if "Source_DB" in df.columns:
        print("\nBy source database:\n" + df["Source_DB"].fillna("(blank)").value_counts().to_string())
    print(f"\nSaved dataset: {args.output_csv}\nSaved log: {args.log_csv}")

if __name__ == "__main__":
    main()
