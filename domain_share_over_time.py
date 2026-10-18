#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot annual technology-domain shares from the three-layer classified CSV.

The script uses fractional counting: each publication's binary technology-domain
indicators are divided by ``tech_n`` before aggregation, then each year's domain
counts are normalised to sum to 100 percent. It writes the annual share table and
high-resolution PDF/PNG figures to the requested output directory.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TECH_COLS = [
    "TECH_D1_AI_and_machine_learning",
    "TECH_D2_Remote_sensing_and_geospatial",
    "TECH_D3_IoT_and_sensor_systems",
    "TECH_D4_Cloud_data_infrastructure_and_FMIS",
    "TECH_D5_Robotics_and_autonomous_machinery",
    "TECH_D6_Decision_support_systems",
    "TECH_D7_Blockchain",
]
LABELS = [
    "AI / ML", "Remote sensing", "IoT & sensors", "Cloud / FMIS",
    "Robotics", "Decision support", "Blockchain",
]
# Okabe-Ito-inspired colour palette retained from the original figure.
COLORS = [
    (0.000, 0.447, 0.698),
    (0.835, 0.369, 0.000),
    (0.000, 0.620, 0.451),
    (0.800, 0.475, 0.655),
    (0.902, 0.624, 0.000),
    (0.337, 0.706, 0.914),
    (0.600, 0.600, 0.600),
]


def calculate_shares(input_csv: Path, year_min: int, year_max: int) -> pd.DataFrame:
    """Load data, apply year range, and return annual percentage shares."""
    header = pd.read_csv(input_csv, nrows=0, encoding="utf-8-sig")
    required = {"Year", "tech_n", *TECH_COLS}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df = pd.read_csv(input_csv, usecols=lambda c: c in required, low_memory=False,
                     encoding="utf-8-sig")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].between(year_min, year_max)].copy()
    if df.empty:
        raise ValueError(f"No records found for years {year_min}–{year_max}.")

    df["Year"] = df["Year"].astype(int)
    for col in TECH_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["tech_n"] = pd.to_numeric(df["tech_n"], errors="coerce").fillna(0)
    denominator = df["tech_n"].replace(0, np.nan)
    fractional = df[TECH_COLS].div(denominator, axis=0)
    yearly_counts = fractional.groupby(df["Year"])[TECH_COLS].sum()
    yearly_totals = yearly_counts.sum(axis=1).replace(0, np.nan)
    yearly_pct = yearly_counts.div(yearly_totals, axis=0).mul(100)
    return yearly_pct.dropna(how="all").fillna(0)


def plot_shares(yearly_pct: pd.DataFrame, output_dir: Path, stem: str, dpi: int = 300) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(8.26, 5.07), dpi=dpi)
    years = yearly_pct.index.to_numpy()
    ax.stackplot(years, [yearly_pct[col].to_numpy() for col in TECH_COLS],
                 labels=LABELS, colors=COLORS, edgecolor="none")
    ax.set_xlim(int(years.min()), int(years.max()))
    ax.set_ylim(0, 100)
    tick_step = 2
    ax.set_xticks(np.arange(int(years.min()), int(years.max()) + 1, tick_step))
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Publication share by technology domain (%)", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=11, length=3.5, width=0.8)
    ax.legend(title="Technology domain", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=10, title_fontsize=10)
    fig.tight_layout()
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    csv_path = output_dir / f"{stem}_data.csv"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    yearly_pct.to_csv(csv_path, index_label="Year", encoding="utf-8-sig")
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"Annual share table: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="Three-layer classified CSV with Year, tech_n and TECH_D1–TECH_D7 columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/domain_share"),
                        help="Output directory (default: outputs/domain_share).")
    parser.add_argument("--year-min", type=int, default=2010, help="First publication year (default: 2010).")
    parser.add_argument("--year-max", type=int, default=2025, help="Last publication year (default: 2025).")
    parser.add_argument("--stem", default="fig_domain_share", help="Output file name stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300 DPI).")
    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f"Input CSV does not exist: {args.input_csv}")
    if args.year_min > args.year_max:
        parser.error("--year-min must be less than or equal to --year-max")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    shares = calculate_shares(args.input_csv, args.year_min, args.year_max)
    plot_shares(shares, args.output_dir, args.stem, args.dpi)


if __name__ == "__main__":
    main()
