#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the full Agriculture 4.0 bibliometric workflow from raw WoS/Scopus exports."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

def run(command):
    print("\n>>>", " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wos-csv", required=True, type=Path, help="Merged WoS export CSV.")
    parser.add_argument("--scopus-articles", required=True, type=Path, help="Scopus article/review export CSV.")
    parser.add_argument("--scopus-conference", required=True, type=Path, help="Scopus conference export CSV.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="Directory for all outputs.")
    parser.add_argument("--skip-cooccurrence", action="store_true", help="Skip co-occurrence network analysis.")
    parser.add_argument("--skip-thematic", action="store_true", help="Skip thematic mapping analysis.")
    parser.add_argument("--keep-in-press", action="store_true", help="Keep in-press records when publication-stage data exist.")
    args = parser.parse_args()
    for p in (args.wos_csv, args.scopus_articles, args.scopus_conference):
        if not p.is_file():
            parser.error(f"Input file does not exist: {p}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    merged = out / "MERGED_DEDUP_STANDARDISED.csv"
    filtered = out / "MERGED_FILTERED.csv"
    log = out / "filter_log.csv"
    run([sys.executable, SRC / "merge_wos_scopus.py", "--wos-csv", args.wos_csv.resolve(),
         "--scopus-articles", args.scopus_articles.resolve(), "--scopus-conference", args.scopus_conference.resolve(),
         "--output-csv", merged])
    filter_cmd = [sys.executable, SRC / "filter_corpus.py", "--input-csv", merged,
                  "--output-csv", filtered, "--log-csv", log]
    if args.keep_in_press:
        filter_cmd.append("--keep-in-press")
    run(filter_cmd)
    if not args.skip_cooccurrence:
        run([sys.executable, SRC / "cooccurrence_network.py", "--input-csv", filtered,
             "--output-dir", out / "cooccurrence"])
    if not args.skip_thematic:
        run([sys.executable, SRC / "thematic_mapping.py", "--input-csv", filtered,
             "--output-dir", out / "thematic_mapping"])
    print(f"\nPipeline completed. Outputs are in: {out}")

if __name__ == "__main__":
    main()
