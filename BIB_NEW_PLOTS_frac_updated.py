# -*- coding: utf-8 -*-
"""
BIB_NEW_PLOTS_frac_updated.py

Main bibliometric figures using the CURRENT three-layer classification file.

IMPORTANT CLASSIFICATION RULE
------------------------------
Technology is represented ONLY by `tech_domains`. A paper carrying several technology domains is exploded into one row per
technology domain whenever an analysis needs a domain-level observation.
No two technology domains are ever treated as one category.

Counting rules
--------------
* Multi-label / integer counts: one publication contributes 1 to EVERY domain
  it carries. Use for "how many publications involve domain X?".
* Fractional counts: a publication carrying k domains contributes 1/k to each
  domain. Use for composition/share figures so domain shares sum to 100%.

The same paper can therefore appear in several technology domains. This is
intentional and consistent with the manuscript's non-mutually-exclusive
classification scheme.

INPUT
-----
`three_layer_classified_FINAL_v4.csv` (current three-layer output)

OUTPUT
------
PNG + PDF figures and CSV tables in `figures_frac_updated/`.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================
DATA_FILENAME = "three_layer_classified_FINAL_v4.csv"
OUT_SUBDIR = "figures_frac_updated"

# Repository layout: data/ contains the input CSV; outputs/ receives generated files.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", DATA_FILENAME)

WRITE = True
ONLY = None                 # e.g. ["F04", "F07"]; None = all
DPI = 300
SHOW_CODES = False

TOP_N_COUNTRIES = 20
TOP_N_MANUSCRIPT = 6
SHOW_REST = True
SHOW_SEGMENT_VALUES = False
SHOW_BAR_TOTALS = True
COUNTRY_MODE = "whole"     # "whole" = every country on a paper; "lead" = first
FIRST_YEAR, LATEST_YEAR = 2010, 2025
NP = r"$N_p$"

# Colour palette retained from the supplied script.
CODE_COLORS = {
    "D1": "#0072B2", "D2": "#D55E00", "D3": "#009E73", "D4": "#CC79A7",
    "D5": "#E69F00", "D6": "#56B4E9", "D7": "#999999", "D8": "#F0E442",
}
CODE_SHORT = {
    "D1": "AI / ML", "D2": "Remote sensing", "D3": "IoT & sensors",
    "D4": "Cloud / FMIS", "D5": "Robotics", "D6": "Decision support",
    "D7": "Blockchain",
}
DOCTYPE_ORDER = ["Article", "Conference paper", "Review"]
DOCTYPE_COLORS = {
    "Article": "#0072B2", "Conference paper": "#E69F00", "Review": "#009E73",
}

# =============================================================================
# PATHS
# =============================================================================
def resolve_infile():
    if os.path.isfile(DATA_PATH):
        return os.path.abspath(DATA_PATH)
    raise FileNotFoundError(
        f"Input CSV not found: {DATA_PATH}\n"
        "Place the classification file in the repository's data/ folder. "
        "See README.md for the expected filename and columns."
    )

INFILE = resolve_infile()
OUTDIR = os.path.join(REPO_ROOT, "outputs", OUT_SUBDIR)

COL_YEAR = "Year"
COL_CITED = "Cited by"
COL_DOCTYPE = "Document Type"
COL_TECH = "tech_domains"
COL_COUNTRIES = "countries"

# =============================================================================
# LABEL HELPERS
# =============================================================================
def domain_code(label):
    m = re.match(r"^(D\d+)\b", str(label))
    return m.group(1) if m else str(label)


def strip_code(label):
    return re.sub(r"^D\d+\s+", "", str(label))


def slug(text):
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(text))).strip("_")


def _labels(cell):
    """Split a semicolon-delimited classification into individual labels."""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [x.strip() for x in str(cell).split(";") if x.strip()]


def explode_labels(df, column, out_col):
    """Explode ONE classification column; never combine categories."""
    d = df[df[column].notna() & (df[column].astype(str).str.strip() != "")].copy()
    d[out_col] = d[column].map(_labels)
    d = d.explode(out_col, ignore_index=False)
    d[out_col] = d[out_col].astype(str).str.strip()
    d = d[d[out_col] != ""].copy()
    d["_paper_id"] = np.arange(len(df))[d.index] if False else d.index
    # The source rows are unique; retain one row per paper/category.
    d = d.reset_index(names="_source_index")
    d = d.drop_duplicates(subset=["_source_index", out_col])
    return d


def resolve_domains(df):
    """All technology domains present in `tech_domains`, ordered by D-number."""
    seen = set()
    for cell in df[COL_TECH].fillna(""):
        seen.update(_labels(cell))

    def sort_key(lbl):
        c = domain_code(lbl)
        m = re.match(r"^D(\d+)$", c)
        return (0, int(m.group(1))) if m else (1, lbl)

    return sorted(seen, key=sort_key)


def build_label_maps(domains):
    colors, short = {}, {}
    fallback = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                "#E69F00", "#56B4E9", "#999999", "#F0E442"]
    for i, d in enumerate(domains):
        c = domain_code(d)
        colors[d] = CODE_COLORS.get(c, fallback[i % len(fallback)])
        s = CODE_SHORT.get(c) or " ".join(strip_code(d).split()[:2])
        short[d] = f"{c} {s}" if SHOW_CODES and c.startswith("D") else s
    return colors, short

# =============================================================================
# LOAD
# =============================================================================
def load_classified(verbose=True):
    df = pd.read_csv(INFILE, low_memory=False, encoding="utf-8-sig")
    df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
    df = df[df[COL_YEAR].notna()].copy()
    df[COL_YEAR] = df[COL_YEAR].astype(int)
    df = df[df[COL_YEAR].between(FIRST_YEAR, LATEST_YEAR)].copy()

    if COL_COUNTRIES not in df.columns and "Countries" in df.columns:
        df[COL_COUNTRIES] = df["Countries"]

    if COL_TECH not in df.columns:
        raise KeyError(f"Required column '{COL_TECH}' is missing from {INFILE}")

    if verbose:
        n_t = df[COL_TECH].notna().sum()
        n_c = df[COL_COUNTRIES].notna().sum() if COL_COUNTRIES in df.columns else 0
        print(f"INPUT  {INFILE}")
        print(f"loaded {len(df):,} records, {FIRST_YEAR}-{LATEST_YEAR}")
        print(f"  with technology domain : {n_t:,} ({100*n_t/len(df):.1f}%)")
        print(f"  with country resolved  : {n_c:,} ({100*n_c/len(df):.1f}%)")
    return df


def classified_only(df, quiet=False):
    d = df[df[COL_TECH].notna() & (df[COL_TECH].astype(str).str.strip() != "")].copy()
    if not quiet:
        n = len(df) - len(d)
        print(f"  {n:,} papers with no technology domain excluded "
              f"({100*n/len(df):.1f}%)")
    return d


def explode_countries(df, mode=COUNTRY_MODE):
    d = df[df[COL_COUNTRIES].notna() & (df[COL_COUNTRIES].astype(str).str.strip() != "")].copy()
    if mode == "lead":
        d["country"] = d[COL_COUNTRIES].astype(str).str.split(";").str[0].str.strip()
        return d
    d["country"] = d[COL_COUNTRIES].astype(str).str.split(";")
    d = d.explode("country")
    d["country"] = d["country"].astype(str).str.strip()
    d = d[d["country"] != ""]
    keys = [c for c in ("DOI", "Title") if c in d.columns] + ["country"]
    return d.drop_duplicates(subset=keys).reset_index(drop=True)


def domain_year_counts(df, domains):
    """Integer multi-label counts: one row per paper-domain assignment."""
    d = explode_labels(classified_only(df, quiet=True), COL_TECH, "domain")
    d = d[d["domain"].isin(domains)]
    out = d.groupby([COL_YEAR, "domain"]).size().unstack(fill_value=0)
    return out.reindex(index=range(FIRST_YEAR, LATEST_YEAR + 1), columns=domains,
                       fill_value=0).astype(int)


def domain_year_fractional(df, domains):
    """Fractional publication equivalents: 1/k for a paper with k domains."""
    d0 = classified_only(df, quiet=True)
    rows = []
    for idx, cell in d0[COL_TECH].items():
        ds = [x for x in _labels(cell) if x in domains]
        if not ds:
            continue
        w = 1.0 / len(set(ds))
        for dom in sorted(set(ds)):
            rows.append((d0.at[idx, COL_YEAR], dom, w))
    if not rows:
        return pd.DataFrame(0.0, index=range(FIRST_YEAR, LATEST_YEAR + 1), columns=domains)
    d = pd.DataFrame(rows, columns=[COL_YEAR, "domain", "weight"])
    out = d.pivot_table(index=COL_YEAR, columns="domain", values="weight",
                        aggfunc="sum", fill_value=0.0)
    return out.reindex(index=range(FIRST_YEAR, LATEST_YEAR + 1), columns=domains,
                       fill_value=0.0)


def country_domain_matrix(df, domains, countries, fractional=True):
    """Country x technology domain using exploded domain assignments."""
    d = explode_countries(classified_only(df, quiet=True))
    d = d[d["country"].isin(countries)].copy()
    rows = []
    for idx, cell in d[COL_TECH].items():
        ds = sorted(set(x for x in _labels(cell) if x in domains))
        if not ds:
            continue
        w = 1.0 / len(ds) if fractional else 1.0
        for dom in ds:
            rows.append((d.at[idx, "country"], dom, w))
    if not rows:
        return pd.DataFrame(0.0, index=countries, columns=domains)
    r = pd.DataFrame(rows, columns=["country", "domain", "weight"])
    mat = r.pivot_table(index="country", columns="domain", values="weight",
                        aggfunc="sum", fill_value=0.0)
    mat = mat.reindex(index=countries, columns=domains, fill_value=0.0)
    return mat if fractional else mat.astype(int)


def doctype_tables(df):
    d = df[df[COL_DOCTYPE].isin(DOCTYPE_ORDER)].copy()
    pubs = (d.groupby([COL_YEAR, COL_DOCTYPE]).size().unstack(fill_value=0)
            .reindex(index=range(FIRST_YEAR, LATEST_YEAR + 1),
                     columns=DOCTYPE_ORDER, fill_value=0))
    cites = (d.groupby([COL_YEAR, COL_DOCTYPE])[COL_CITED].sum()
             .unstack(fill_value=0)
             .reindex(index=range(FIRST_YEAR, LATEST_YEAR + 1),
                      columns=DOCTYPE_ORDER, fill_value=0))
    return pubs, cites


def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)


def save(fig, name):
    if not WRITE:
        plt.close(fig)
        return
    ensure_outdir()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=DPI,
                    bbox_inches="tight")
    print(f"  wrote {name}.png / .pdf")


def save_csv(obj, name):
    if not WRITE:
        return
    ensure_outdir()
    obj.to_csv(os.path.join(OUTDIR, name))
    print(f"  wrote {name}")


def tidy(ax, xt=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if xt is not None:
        ax.set_xticks(xt)

# =============================================================================
# FIGURES
# =============================================================================
def F01_publication_growth(df, D):
    print("\nF01  publication growth by document type")
    pubs, _ = doctype_tables(df)
    years = pubs.index.values
    total = pubs.sum(axis=1).values.astype(float)
    n0, n1 = total[0], total[-1]
    span = years[-1] - years[0]
    cagr = (n1 / n0) ** (1 / span) - 1
    print(f"  CAGR {years[0]}-{years[-1]}: {100*cagr:.1f}% p.a. "
          f"({int(n0):,} -> {int(n1):,})")

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(years))
    for dt in DOCTYPE_ORDER:
        v = pubs[dt].values.astype(float)
        ax.bar(years, v, bottom=bottom, width=0.78,
               color=DOCTYPE_COLORS[dt], label=dt)
        bottom += v
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(NP, fontsize=14)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    tidy(ax, np.arange(years.min(), years.max() + 1, 2))
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    save(fig, "fig_publication_growth")
    save_csv(pubs.assign(total=total, cagr_curve=(n0*(1+cagr)**(years-years[0])).round(1)),
             "publication_growth_by_doctype.csv")
    return fig


def F02_citation_growth(df, D):
    print("\nF02  citation growth by document type")
    _, cites = doctype_tables(df)
    years = cites.index.values
    fig, ax = plt.subplots(figsize=(8, 5))
    for dt in DOCTYPE_ORDER:
        ax.plot(years, cites[dt].values, marker="o", markersize=3.5,
                linewidth=1.5, label=dt, color=DOCTYPE_COLORS[dt])
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Citation count", fontsize=14)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    tidy(ax, np.arange(years.min(), years.max() + 1, 2))
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    save(fig, "fig_citation_growth")
    save_csv(cites, "citation_growth_by_doctype.csv")
    return fig


def F03_cumulative_citations(df, D):
    print("\nF03  cumulative citations")
    _, cites = doctype_tables(df)
    cum = cites.sum(axis=1).cumsum()
    years = cum.index.values
    print(f"  cumulative citations: {int(cum.iloc[-1]):,}")
    fig, ax = plt.subplots(figsize=(8, 5))
    v = cum.values.astype(float)
    ax.fill_between(years, v, alpha=0.16)
    ax.plot(years, v, marker="o", markersize=4, linewidth=1.8)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Cumulative citation count", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1000:,.0f}k" if x >= 1000 else f"{x:,.0f}"))
    tidy(ax, np.arange(years.min(), years.max() + 1, 2))
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    save(fig, "fig_cumulative_citations")
    save_csv(cum.rename("cumulative_citations"), "cumulative_citations.csv")
    return fig


def F04_domain_share(df, D):
    print("\nF04  technology-domain composition (fractional counting)")
    counts = domain_year_fractional(df, D["order"])
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    print("  first year -> last year:")
    for dom in D["order"]:
        print(f"    {D['short'][dom]:<18} {shares.loc[FIRST_YEAR, dom]:5.1f}% -> "
              f"{shares.loc[LATEST_YEAR, dom]:5.1f}%")
    years = shares.index.values
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.stackplot(years, [shares[d].values for d in D["order"]],
                 labels=[D["short"][d] for d in D["order"]],
                 colors=[D["colors"][d] for d in D["order"]], alpha=0.92)
    ax.set_xlabel("Year")
    ax.set_ylabel("Publication share by technology domain (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(years.min(), years.max())
    ax.set_xticks(np.arange(years.min(), years.max() + 1, 2))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=12,
              frameon=False, title="Technology domain", title_fontsize=12)
    fig.tight_layout()
    save(fig, "fig_domain_share")
    save_csv(shares.round(2), "domain_share.csv")
    return fig


def F05_domain_counts_log(df, D):
    print("\nF05  absolute technology-domain counts, log scale")
    counts = domain_year_counts(df, D["order"])
    years = counts.index.values
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for dom in D["order"]:
        v = counts[dom].values.astype(float)
        ax.plot(years, np.where(v > 0, v, np.nan), color=D["colors"][dom],
                marker="o", markersize=3.5, linewidth=1.6,
                label=D["short"][dom])
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel(NP + "  (log scale)")
    ax.set_xticks(np.arange(years.min(), years.max() + 1, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=12,
              frameon=False, title="Technology domain", title_fontsize=12)
    fig.tight_layout()
    save(fig, "fig_domain_counts_log")
    save_csv(counts, "domain_counts.csv")
    return fig


def F06_domain_growth(df, D):
    print("\nF06  growth of each technology domain")
    counts = domain_year_counts(df, D["order"])
    years = counts.index.values
    out = {}
    for dom in D["order"]:
        v = counts[dom].values.astype(float)
        a, b = int(counts.loc[FIRST_YEAR, dom]), int(counts.loc[LATEST_YEAR, dom])
        cg = ((b/a)**(1/(LATEST_YEAR-FIRST_YEAR))-1)*100 if a > 0 else np.nan
        print(f"    {D['short'][dom]:<18} {a:,} -> {b:,}   CAGR "
              f"{cg:.1f}%" if np.isfinite(cg) else
              f"    {D['short'][dom]:<18} {a:,} -> {b:,}   CAGR n/a")
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.fill_between(years, v, color=D["colors"][dom], alpha=0.16)
        ax.plot(years, v, color=D["colors"][dom], marker="o", markersize=4,
                linewidth=1.9)
        ax.set_xlabel("Year")
        ax.set_ylabel(NP)
        ax.set_ylim(bottom=0)
        ax.annotate(D["short"][dom], xy=(0.03, 0.95), xycoords="axes fraction",
                    ha="left", va="top", fontsize=12, color="black",
                    fontweight="bold")
        tidy(ax, np.arange(FIRST_YEAR, LATEST_YEAR + 1, 3))
        fig.tight_layout()
        name = f"fig_domain_growth_{domain_code(dom)}_{slug(D['short'][dom])}"
        save(fig, name)
        out[dom] = fig
    save_csv(counts, "domain_growth_counts.csv")
    return out


def F07_country_domain_share(df, D):
    """
    F07 -- country composition across technology domains.

    Each country's bar is normalized to 100%, so the figure compares
    TECHNOLOGICAL COMPOSITION rather than absolute country size.

    Counting:
        - Technology domains: fractional counting.
        - A paper with k technology domains contributes 1/k to each domain.
        - Countries are handled according to COUNTRY_MODE.

    Right-side labels:
        - Total fractional publication equivalents are shown at the end
          of each bar.
        - A $N_p$ header is placed above these values.

    Field-average boundaries:
        - Calculated from the overall fractional technology-domain composition
          of the displayed countries.
        - Vertical dashed lines show cumulative field-average shares.
        - Boundaries are shown only when the corresponding domain contributes
          at least FIELD_AVERAGE_MIN_SHARE percent of the field, avoiding
          excessive clustering of lines.
    """

    print("\nF07  country composition across technology domains")

    # -------------------------------------------------------------------------
    # SETTINGS FOR THIS FIGURE
    # -------------------------------------------------------------------------
    SHOW_FIELD_AVERAGE_BOUNDARIES = True
    FIELD_AVERAGE_MIN_SHARE = 5.0

    # -------------------------------------------------------------------------
    # COUNTRY DATA
    # -------------------------------------------------------------------------
    d = explode_countries(classified_only(df))

    print(f"  country counting: {COUNTRY_MODE.upper()}")

    # Top countries by number of country-paper assignments
    top = (
        d["country"]
        .value_counts()
        .head(TOP_N_COUNTRIES)
        .index
        .tolist()
    )

    # -------------------------------------------------------------------------
    # COUNTRY x TECHNOLOGY-DOMAIN MATRIX
    #
    # Fractional counting:
    #   k domains -> 1/k to each domain.
    #
    # This preserves the non-mutually-exclusive classification scheme.
    # -------------------------------------------------------------------------
    mat = country_domain_matrix(
        df,
        D["order"],
        top,
        fractional=True
    )

    # Sort countries by total fractional output
    mat = mat.loc[
        mat.sum(axis=1)
        .sort_values(ascending=False)
        .index
    ]

    # -------------------------------------------------------------------------
    # CONVERT TO WITHIN-COUNTRY SHARES
    # -------------------------------------------------------------------------
    share = mat.div(
        mat.sum(axis=1),
        axis=0
    ) * 100

    # -------------------------------------------------------------------------
    # FIELD-AVERAGE TECHNOLOGY COMPOSITION
    #
    # This is calculated across the displayed countries using the fractional
    # publication-equivalent matrix.
    #
    # Example:
    #   D1 = 35%
    #   D2 = 25%
    #   D3 = 15%
    #
    # Then vertical boundaries are drawn at:
    #   35%, 60%, 75%, ...
    # -------------------------------------------------------------------------
    world = (
        mat.sum(axis=0)
        / mat.values.sum()
        * 100
    )

    print("\n  share of national output by domain (%):")
    print(
        share.round(1)
        .rename(columns=D["short"])
        .to_string()
    )

    print("\n  field-average composition (%):")
    for dom in D["order"]:
        print(
            f"    {D['short'][dom]:<20} "
            f"{world[dom]:5.1f}%"
        )

    # -------------------------------------------------------------------------
    # REVERSE ORDER FOR HORIZONTAL BAR PLOT
    # -------------------------------------------------------------------------
    s = share.iloc[::-1]
    tot = mat.sum(axis=1).iloc[::-1]

    y = np.arange(len(s))

    # -------------------------------------------------------------------------
    # FIGURE
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(12, 8.5)
    )

    # -------------------------------------------------------------------------
    # STACKED 100% BARS
    # -------------------------------------------------------------------------
    left = np.zeros(len(s))

    for dom in D["order"]:

        v = s[dom].values.astype(float)

        ax.barh(
            y,
            v,
            left=left,
            height=0.72,
            color=D["colors"][dom],
            label=D["short"][dom],
            edgecolor="white",
            linewidth=0.6,
            zorder=2
        )

        # Optional values inside the segments
        if SHOW_SEGMENT_VALUES:

            for yi, (val, l) in enumerate(
                zip(v, left)
            ):

                if val >= 7:

                    ax.text(
                        l + val / 2,
                        yi,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                        fontweight="bold",
                        zorder=4
                    )

        left += v

    # -------------------------------------------------------------------------
    # Np HEADER + TOTALS
    #
    # The bars themselves are normalized to 100%, so the actual publication
    # equivalent is shown separately at the right.
    # -------------------------------------------------------------------------
    if SHOW_BAR_TOTALS:

        # Header
        ax.text(
            104.5,
            len(s) - 0.15,
            NP,
            fontsize=12,
            ha="center",
            va="center",
            color="#555555"
        )

        # Country totals
        for yi, t in zip(y, tot.values):

            ax.text(
                101.2,
                yi,
                f"{int(round(t)):,}",
                va="center",
                ha="left",
                fontsize=12,
                color="black"
            )

    # -------------------------------------------------------------------------
    # FIELD-AVERAGE BOUNDARIES
    #
    # These reproduce the logic from the previous version:
    #
    #   cumulative field-average share
    #
    # A boundary is drawn after each sufficiently large domain.
    # -------------------------------------------------------------------------
    if SHOW_FIELD_AVERAGE_BOUNDARIES:

        cumulative = 0.0

        for dom in D["order"][:-1]:

            cumulative += world[dom]

            if world[dom] >= FIELD_AVERAGE_MIN_SHARE:

                ax.axvline(
                    cumulative,
                    color="#333333",
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.6,
                    zorder=5
                )

    # -------------------------------------------------------------------------
    # AXES
    # -------------------------------------------------------------------------
    ax.set_yticks(y)

    ax.set_yticklabels(
        s.index,
        fontsize=12
    )

    ax.set_ylabel(
        "Country",
        fontsize=12
    )

    # Leave room on the right for Np values
    ax.set_xlim(
        0,
        108
    )

    ax.set_xticks(
        np.arange(0, 101, 12)
    )

    ax.set_xlabel(
        "Share of the country's Agriculture 4.0 output (%)",
        fontsize=12
    )

    ax.set_ylim(
        -0.7,
        len(s) - 0.3
    )

    # -------------------------------------------------------------------------
    # STYLE
    # -------------------------------------------------------------------------
    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0
    )

    # -------------------------------------------------------------------------
    # LEGEND
    #
    # Add a dummy handle for the field-average boundary so the dashed lines
    # are explained in the figure.
    # -------------------------------------------------------------------------
    if SHOW_FIELD_AVERAGE_BOUNDARIES:

        ax.plot(
            [],
            [],
            color="#333333",
            linestyle="--",
            linewidth=0.8,
            label="Field-average boundary"
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=12,
        frameon=False
    )

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    save(
        fig,
        "fig_country_domain_share"
    )

    # -------------------------------------------------------------------------
    # EXPORT TABLES
    # -------------------------------------------------------------------------
    save_csv(
        mat,
        "country_domain_counts_fractional.csv"
    )

    save_csv(
        share.round(2),
        "country_domain_share.csv"
    )

    # Also export the field-average composition
    field_average = pd.DataFrame(
        {
            "domain": [
                D["short"][d]
                for d in D["order"]
            ],
            "fractional_share_pct": [
                round(float(world[d]), 3)
                for d in D["order"]
            ]
        }
    )

    save_csv(
        field_average,
        "country_domain_field_average.csv"
    )

    return fig


def F08_domain_trends(df, D):
    print("\nF08  technology-domain trends")
    counts = domain_year_counts(df, D["order"])
    years = counts.index.values
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b",
                   "#e377c2", "#7f7f7f", "#bcbd22"]
    for i, dom in enumerate(D["order"]):
        ax.plot(years, counts[dom].values.astype(float),
                color=line_colors[i % len(line_colors)], linestyle="--",
                linewidth=1.0, marker="o", markersize=3.2,
                markeredgewidth=0, label=D["short"][dom])
    ax.set_xlabel("Year")
    ax.set_ylabel(NP, fontsize=12)
    ax.set_xticks(np.arange(years.min(), years.max()+1, 2))
    ax.set_xlim(years.min()-0.5, years.max()+0.5)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", handlelength=2.4, labelspacing=0.3,
              borderaxespad=0.6)
    fig.tight_layout()
    save(fig, "fig_domain_trends")
    return fig


def F09_domain_country_bars(df, D):
    print("\nF09  technology domain x country bars")
    d = explode_countries(classified_only(df, quiet=True))
    totals = d["country"].value_counts()
    top = totals.head(TOP_N_MANUSCRIPT).index.tolist()
    d["_country_plot"] = np.where(d["country"].isin(top), d["country"], "Other countries")
    cols = top + (["Other countries"] if SHOW_REST else [])

    # Explicitly explode technology domains. One domain per row; a paper with
    # three domains contributes once to each of its three domain-country cells.
    rows = []
    for idx, cell in d[COL_TECH].items():
        for dom in sorted(set(_labels(cell))):
            if dom in D["order"]:
                rows.append((dom, d.at[idx, "_country_plot"]))
    x = pd.DataFrame(rows, columns=["domain", "country"])
    mat = x.groupby(["domain", "country"]).size().unstack(fill_value=0)
    mat = mat.reindex(index=D["order"], columns=cols, fill_value=0).astype(int)
    order = mat.sum(axis=1).sort_values(ascending=True).index
    m = mat.loc[order]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(m))
    palette = plt.cm.viridis(np.linspace(0.05, 0.9, len(top)))
    colors = {c: palette[i] for i, c in enumerate(top)}
    colors["Other countries"] = "#c8c8c8"
    left = np.zeros(len(m))
    for c in m.columns:
        v = m[c].values.astype(float)
        ax.barh(y, v, left=left, height=0.72, color=colors[c], label=c, linewidth=0)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([D["short"][x] for x in m.index])
    ax.set_xlabel(NP, fontsize=11)
    ax.set_ylim(-0.6, len(m)-0.4)
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", title="Countries", title_fontsize=9,
              handlelength=1.2, labelspacing=0.3, borderpad=0.6)
    fig.tight_layout()
    save(fig, "fig_domain_country_bars")
    save_csv(mat, "domain_country_counts.csv")
    return fig


def F10_counting_sensitivity(df, D):
    """Sensitivity of domain composition to integer vs fractional counting."""
    print("\nF10  integer vs fractional technology-domain counting")
    integer = domain_year_counts(df, D["order"])
    frac = domain_year_fractional(df, D["order"])
    rows = []
    for yr in integer.index:
        n = len(df[df[COL_YEAR] == yr])
        for dom in D["order"]:
            rows.append({
                "year": int(yr),
                "domain": D["short"][dom],
                "integer_multilabel_count": int(integer.loc[yr, dom]),
                "fractional_publication_equivalent": round(float(frac.loc[yr, dom]), 3),
                "integer_share_of_assignments_pct": round(
                    100*integer.loc[yr, dom]/integer.loc[yr].sum(), 3),
                "fractional_share_pct": round(
                    100*frac.loc[yr, dom]/frac.loc[yr].sum(), 3),
                "corpus_n": int(n),
            })
    t = pd.DataFrame(rows)
    print(t[t.year.isin([FIRST_YEAR, LATEST_YEAR])].to_string(index=False))
    save_csv(t, "technology_counting_sensitivity.csv")
    return t

# =============================================================================
# MAIN
# =============================================================================
def main():
    df = load_classified()
    domains = resolve_domains(df)
    colors, short = build_label_maps(domains)
    D = {"order": domains, "colors": colors, "short": short}
    print("\nTECHNOLOGY DOMAINS USED (from tech_domains only):")
    for d in domains:
        print(f"  {d}")

    funcs = {
        "F01": F01_publication_growth,
        "F02": F02_citation_growth,
        "F03": F03_cumulative_citations,
        "F04": F04_domain_share,
        "F05": F05_domain_counts_log,
        "F06": F06_domain_growth,
        "F07": F07_country_domain_share,
        "F08": F08_domain_trends,
        "F09": F09_domain_country_bars,
        "F10": F10_counting_sensitivity,
    }
    wanted = funcs.keys() if ONLY is None else ONLY
    for key in wanted:
        if key not in funcs:
            raise KeyError(f"Unknown figure key: {key}")
        funcs[key](df, D)

    print("\nDONE")
    print(f"Input : {INFILE}")
    print(f"Output: {OUTDIR}")


if __name__ == "__main__":
    main()
