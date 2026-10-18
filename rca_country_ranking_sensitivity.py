# -*- coding: utf-8 -*-
"""
================================================================================
 RCA-BASED COUNTRY RANKING PIPELINE  (Agriculture 4.0 technology domains)  v5
================================================================================

Input : three_layer_classified_FINAL_v4.csv  (one row = one publication)
Output: outputsRCA/*.csv  and  figuresRCA/*.png

Ranks countries by REVEALED COMPARATIVE ADVANTAGE across seven Agriculture 4.0
technology domains, using the two source columns `tech_domains` and `countries`.

WHAT THIS SCRIPT DOES
---------------------
1.  Explode `countries` (';' separated)  -> one country / row
2.  Explode `tech_domains` (';' separated)-> one domain  / row
3.  Strip / normalise whitespace on both; keep the seven canonical domains.
4.  Fractional counting: a paper with n countries gives each country 1/n.
5.  Split 2010-2025 into 4 equal windows + a pooled ALL segment.
6.  Build a country x technology matrix per window and compute the Balassa RCA.
7.  Apply a stationary per-cell publication floor (reliability mask).
8.  Rank countries by a parameter-free composite = geometric mean of a
    distinctiveness (RSCA x inverse-ubiquity) score and an activity score.
    A plain output-volume rank is reported ALONGSIDE for interpretation.
9.  Sensitivity analysis:
      (a) sweep RCA cut-off      (b) sweep publication threshold
      (c) full 2-D cut-off x threshold grid
      (d) 14-variant one-at-a-time robustness (counting, aggregation, ...)
    All agreement measured by Spearman rho, Kendall tau, top-N Jaccard.

This is the cleaned/consolidated successor to rca_ranking_pipelineO.py; the
ranking methodology is unchanged (RCA specialisation, as requested).
"""

import argparse
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

warnings.filterwarnings("ignore")

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================
INFILE = None
OUTDIR = None
FIGDIR = None

COL_YEAR      = "Year"
COL_COUNTRIES = "countries"
COL_DOMAINS   = "tech_domains"

PERIODS = [(2010, 2013), (2014, 2017), (2018, 2021), (2022, 2025)]

DOMAIN_ORDER = [
    "D1 AI & machine learning",
    "D2 Remote sensing & geospatial",
    "D3 IoT & sensor systems",
    "D4 Cloud, data infrastructure & FMIS",
    "D5 Robotics & autonomous machinery",
    "D6 Decision support systems",
    "D7 Blockchain",
]

# --- baseline (headline) parameters ------------------------------------------
RCA_CUTOFF_BASE   = 1
PUB_THRESH_MODE   = "global"
PUB_THRESH_PCT    = 25
PUB_THRESH_FLOOR  = 5
THRESHOLD_SCOPE   = "pooled"        # stationary threshold from the pooled corpus
MIN_COUNTRY_OUT   = 5

# --- sensitivity grids -------------------------------------------------------
RCA_CUTOFF_GRID   = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
PUB_THRESH_GRID   = [1, 2, 3, 5, 10]


# =============================================================================
# 1.  LOAD + EXPLODE + CLEAN + FRACTIONAL COUNTING
# =============================================================================
def load_long(infile=None):
    if infile is None:
        infile = INFILE
    if infile is None:
        raise ValueError("Input CSV was not configured. Pass --input-csv.")
    df = pd.read_csv(infile, low_memory=False, encoding="utf-8-sig",
                     usecols=lambda c: c in {COL_YEAR, COL_COUNTRIES,
                                             COL_DOMAINS, "Countries"})

    df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
    df = df[df[COL_YEAR].notna()].copy()
    df[COL_YEAR] = df[COL_YEAR].astype(int)
    df = df[(df[COL_YEAR] >= 2010) & (df[COL_YEAR] <= 2025)].copy()

    if COL_COUNTRIES not in df.columns and "Countries" in df.columns:
        df[COL_COUNTRIES] = df["Countries"]

    df = df[df[COL_COUNTRIES].notna() & df[COL_DOMAINS].notna()].copy()
    df = df[(df[COL_COUNTRIES].astype(str).str.strip() != "") &
            (df[COL_DOMAINS].astype(str).str.strip() != "")].copy()

    df["pid"] = np.arange(len(df))

    # explode countries
    df["country"] = df[COL_COUNTRIES].astype(str).str.split(";")
    df = df.explode("country")
    df["country"] = (df["country"].str.strip()
                                   .str.replace(r"\s+", " ", regex=True))
    df = df[df["country"].str.len() > 2]
    df = df.drop_duplicates(subset=["pid", "country"])

    # fractional weight = 1 / distinct countries on the paper
    n_ctry = df.groupby("pid")["country"].transform("nunique")
    df["n_countries"] = n_ctry
    df["w_frac"] = 1.0 / n_ctry

    # explode domains
    df["domain"] = df[COL_DOMAINS].astype(str).str.split(";")
    df = df.explode("domain")
    df["domain"] = (df["domain"].str.strip()
                                 .str.replace(r"\s+", " ", regex=True))
    df = df[df["domain"].isin(DOMAIN_ORDER)]
    df = df.drop_duplicates(subset=["pid", "country", "domain"])

    def to_period(y):
        for a, b in PERIODS:
            if a <= y <= b:
                return f"{a}-{b}"
        return None
    df["period"] = df[COL_YEAR].map(to_period)

    keep = ["pid", COL_YEAR, "period", "country", "domain",
            "n_countries", "w_frac"]
    return df[keep].reset_index(drop=True)


# =============================================================================
# 2.  COUNT MATRICES
# =============================================================================
def build_matrices(long):
    X_frac = (long.groupby(["country", "domain"])["w_frac"].sum()
                  .unstack(fill_value=0.0)
                  .reindex(columns=DOMAIN_ORDER, fill_value=0.0))
    X_whole = (long.groupby(["country", "domain"])["pid"].nunique()
                   .unstack(fill_value=0)
                   .reindex(columns=DOMAIN_ORDER, fill_value=0)
                   .reindex(index=X_frac.index, fill_value=0))
    return X_frac, X_whole


# =============================================================================
# 3.  BALASSA RCA
# =============================================================================
def compute_rca(X):
    X = X.astype(float)
    row = np.array(X.sum(axis=1).values, dtype=float).reshape(-1, 1)
    col = np.array(X.sum(axis=0).values, dtype=float).reshape(1, -1)
    tot = float(X.values.sum())
    row[row == 0] = np.nan
    col[col == 0] = np.nan
    rca = (X.values / row) / (col / tot)
    return pd.DataFrame(np.nan_to_num(rca), index=X.index, columns=X.columns)


# =============================================================================
# 4.  PUBLICATION THRESHOLD
# =============================================================================
def tech_thresholds(X_whole, mode=PUB_THRESH_MODE,
                    pct=PUB_THRESH_PCT, floor=PUB_THRESH_FLOOR,
                    global_floor=None):
    tau = {}
    for d in DOMAIN_ORDER:
        if mode == "global":
            tau[d] = int(global_floor if global_floor is not None else floor)
        else:
            pos = X_whole[d][X_whole[d] > 0]
            q = np.percentile(pos, pct) if len(pos) else 0
            tau[d] = int(max(floor, np.ceil(q)))
    return tau


def binarize(rca, X_whole, cutoff, tau):
    thr = pd.Series(tau).reindex(DOMAIN_ORDER)
    passes_pub = X_whole.ge(thr, axis=1)
    return ((rca >= cutoff) & passes_pub).astype(int)


# =============================================================================
# 5.  RANKING
# =============================================================================
def rsca_matrix(rca):
    return (rca - 1.0) / (rca + 1.0)


def _mm(s):
    s = s.astype(float); r = s.max() - s.min()
    return (s - s.min()) / r if r > 0 else s * 0.0 + 0.5


def rank_countries(long, X_frac, X_whole, rca, cutoff, tau,
                   min_output=MIN_COUNTRY_OUT):
    M = binarize(rca, X_whole, cutoff, tau)
    rsca = rsca_matrix(rca)

    thr = pd.Series(tau).reindex(DOMAIN_ORDER)
    reliable = X_whole.ge(thr, axis=1)

    tot_whole = X_whole.sum(axis=1)
    elig = tot_whole[tot_whole >= min_output].index
    Me = M.loc[elig]
    Re = reliable.loc[elig]

    C = max(len(elig), 1)
    ubiq = Me.sum(axis=0).replace(0, np.nan)
    idf = np.log(C / ubiq).reindex(DOMAIN_ORDER).fillna(0.0)

    out = pd.DataFrame(index=elig)
    out["papers_whole"]      = tot_whole.loc[elig]
    out["papers_fractional"] = X_frac.sum(axis=1).loc[elig]
    out["breadth"]           = Me.sum(axis=1)
    out["rsca_strength"]     = (rsca.loc[elig].clip(lower=0) * Re).sum(axis=1)

    distinct = (rsca.loc[elig].clip(lower=0) * Re).mul(idf, axis=1).sum(axis=1)
    out["distinctiveness"] = distinct
    activity = np.log1p(tot_whole.loc[elig].astype(float))

    out["spec_score"]     = _mm(distinct)
    out["activity_score"] = _mm(activity)
    out["composite"]      = np.sqrt(out["spec_score"] * out["activity_score"])
    out["composite_rank"] = out["composite"].rank(ascending=False,
                                                   method="min").astype(int)

    # plain output-volume rank, reported ALONGSIDE the composite so the
    # specialisation ranking can be read against raw research output.
    out["volume_rank"] = out["papers_whole"].rank(ascending=False,
                                                   method="min").astype(int)

    for c in ["distinctiveness", "activity_score", "breadth", "rsca_strength"]:
        out[c + "_rank"] = out[c].rank(ascending=False, method="min").astype(int)

    out["TOPSIS"] = out["composite"]
    out["TOPSIS_rank"] = out["composite_rank"]
    out["consensus_rank"] = out["composite_rank"]
    return out.sort_values("composite_rank")


# =============================================================================
# 6.  SENSITIVITY  (cut-off sweep, threshold sweep, 2-D grid)
# =============================================================================
def _ranking_vector(long, X_frac, X_whole, rca, cutoff, tau,
                    metric="TOPSIS", min_output=MIN_COUNTRY_OUT):
    tbl = rank_countries(long, X_frac, X_whole, rca, cutoff, tau, min_output)
    return tbl[metric + "_rank"]


def sensitivity(long, X_frac, X_whole, rca, metric="TOPSIS", topn=10,
                base_tau=None):
    if base_tau is None:
        base_tau = tech_thresholds(X_whole, mode=PUB_THRESH_MODE)
    base_rank = _ranking_vector(long, X_frac, X_whole, rca,
                                RCA_CUTOFF_BASE, base_tau, metric)
    base_top  = set(base_rank.sort_values().head(topn).index)

    def compare(r):
        common = base_rank.index.intersection(r.index)
        if len(common) < 3:
            return np.nan, np.nan, np.nan
        rho = spearmanr(base_rank[common], r[common]).correlation
        tau = kendalltau(base_rank[common], r[common]).correlation
        top = set(r.sort_values().head(topn).index)
        jac = len(base_top & top) / len(base_top | top)
        return rho, tau, jac

    rows = []
    for cut in RCA_CUTOFF_GRID:
        r = _ranking_vector(long, X_frac, X_whole, rca, cut, base_tau, metric)
        rho, tau, jac = compare(r)
        rows.append(dict(rca_cutoff=cut, spearman=rho, kendall=tau,
                         top_overlap=jac))
    cut_df = pd.DataFrame(rows)

    rows = []
    for thr in PUB_THRESH_GRID:
        tau_d = tech_thresholds(X_whole, mode="global", global_floor=thr)
        r = _ranking_vector(long, X_frac, X_whole, rca,
                            RCA_CUTOFF_BASE, tau_d, metric)
        rho, tau, jac = compare(r)
        rows.append(dict(pub_threshold=thr, spearman=rho, kendall=tau,
                         top_overlap=jac))
    thr_df = pd.DataFrame(rows)

    rows = []
    for cut in RCA_CUTOFF_GRID:
        for thr in PUB_THRESH_GRID:
            tau_d = tech_thresholds(X_whole, mode="global", global_floor=thr)
            r = _ranking_vector(long, X_frac, X_whole, rca, cut, tau_d, metric)
            rho, _, jac = compare(r)
            rows.append(dict(rca_cutoff=cut, pub_threshold=thr,
                             spearman=rho, top_overlap=jac))
    grid_df = pd.DataFrame(rows)
    return cut_df, thr_df, grid_df


def save_country_rank_changes(
    long, X_frac, X_whole, rca, period_name, base_tau
):
    """
    Export country-level rank changes for RCA-cutoff and publication-
    threshold sensitivity analyses.

    rank_change = scenario_rank - baseline_rank
    Negative: country improves its rank.
    Positive: country falls in the ranking.
    Zero: rank unchanged.
    """

    os.makedirs(OUTDIR, exist_ok=True)

    # Baseline: theta = 1.0, tau = 5
    baseline_cutoff = 1.0
    baseline_threshold = 5

    baseline_tau = tech_thresholds(
        X_whole, mode="global",
        global_floor=baseline_threshold
    )

    baseline = rank_countries(
        long, X_frac, X_whole, rca,
        cutoff=baseline_cutoff,
        tau=baseline_tau
    )

    baseline_rank = baseline["composite_rank"].rename(
        "baseline_rank"
    )

    all_changes = []

    def compare_scenario(scenario_type, parameter, cutoff, tau_dict):
        scenario = rank_countries(
            long, X_frac, X_whole, rca,
            cutoff=cutoff,
            tau=tau_dict
        )

        scenario_rank = scenario["composite_rank"].rename(
            "scenario_rank"
        )

        # Compare only countries present in both rankings
        comparison = pd.concat(
            [baseline_rank, scenario_rank], axis=1, join="inner"
        ).dropna()

        comparison["country"] = comparison.index
        comparison["scenario_type"] = scenario_type
        comparison["parameter_value"] = parameter
        comparison["baseline_cutoff"] = baseline_cutoff
        comparison["baseline_threshold"] = baseline_threshold

        comparison["rank_change"] = (
            comparison["scenario_rank"]
            - comparison["baseline_rank"]
        )

        comparison["movement"] = np.select(
            [
                comparison["rank_change"] < 0,
                comparison["rank_change"] > 0
            ],
            [
                "Improved",
                "Declined"
            ],
            default="Unchanged"
        )

        all_changes.append(
            comparison.reset_index(drop=True)
        )

    # A. Vary RCA cut-off; keep publication threshold at 5
    for cutoff in RCA_CUTOFF_GRID:
        tau_dict = tech_thresholds(
            X_whole, mode="global",
            global_floor=baseline_threshold
        )

        compare_scenario(
            scenario_type="RCA_cutoff",
            parameter=cutoff,
            cutoff=cutoff,
            tau_dict=tau_dict
        )

    # B. Vary publication threshold; keep RCA cut-off at 1
    for threshold in PUB_THRESH_GRID:
        tau_dict = tech_thresholds(
            X_whole, mode="global",
            global_floor=threshold
        )

        compare_scenario(
            scenario_type="publication_threshold",
            parameter=threshold,
            cutoff=baseline_cutoff,
            tau_dict=tau_dict
        )

    result = pd.concat(all_changes, ignore_index=True)

    result = result[
        [
            "scenario_type",
            "parameter_value",
            "country",
            "baseline_rank",
            "scenario_rank",
            "rank_change",
            "movement",
            "baseline_cutoff",
            "baseline_threshold"
        ]
    ].sort_values(
        ["scenario_type", "parameter_value", "scenario_rank"]
    )

    outpath = os.path.join(
        OUTDIR,
        f"country_rank_changes_{period_name.replace(' ', '_')}.csv"
    )

    result.to_csv(outpath, index=False)

    print(f"Country-level sensitivity ranks saved: {outpath}")

    return result

# =============================================================================
# 6b. IN-DEPTH 14-VARIANT SENSITIVITY  (one discretionary choice at a time)
# =============================================================================
def rank_variant(seg, Xw_all, counting="frac", cutoff=1.0, thr=5,
                 scope="global", agg="geom", min_output=5):
    Xf, Xw = build_matrices(seg)
    Xrca = Xf if counting == "frac" else Xw
    rca  = compute_rca(Xrca); rsca = rsca_matrix(rca)
    if scope == "global":
        tau = {d: thr for d in DOMAIN_ORDER}
    else:
        tau = tech_thresholds(Xw_all, mode="per_tech_percentile",
                              pct=25, floor=thr)
    thrS = pd.Series(tau).reindex(DOMAIN_ORDER)
    tot  = Xw.sum(axis=1); elig = tot[tot >= min_output].index
    R    = Xw.ge(thrS, axis=1).loc[elig]
    M    = ((rca >= cutoff) & Xw.ge(thrS, axis=1)).astype(int).loc[elig]
    C    = max(len(elig), 1); u = M.sum(0).replace(0, np.nan)
    idf  = np.log(C / u).reindex(DOMAIN_ORDER).fillna(0.0)
    distinct = (rsca.loc[elig].clip(lower=0) * R).mul(idf, axis=1).sum(1)
    activity = np.log1p(tot.loc[elig].astype(float))
    spec, act = _mm(distinct), _mm(activity)
    if   agg == "geom":     score = np.sqrt(spec * act)
    elif agg == "arith":    score = 0.5 * (spec + act)
    elif agg == "distinct": score = distinct
    else:                   score = spec
    return score.rank(ascending=False, method="min").astype(int)


def indepth_sensitivity(long):
    WIN = [f"{a}-{b}" for a, b in PERIODS]
    Xf_all, Xw_all = build_matrices(long)

    def compare(base, r, topn):
        common = base.index.intersection(r.index)
        if len(common) < 3:
            return np.nan, np.nan, np.nan
        rho = spearmanr(base[common], r[common]).correlation
        tau = kendalltau(base[common], r[common]).correlation
        bt = set(base.sort_values().head(topn).index)
        rt = set(r.sort_values().head(topn).index)
        jac = len(bt & rt) / len(bt | rt)
        return rho, tau, jac

    BASE = {w: rank_variant(long[long.period == w], Xw_all) for w in WIN}

    VARIANTS = [
        ("RCA cutoff = 0.50",         dict(cutoff=0.50)),
        ("RCA cutoff = 0.75",         dict(cutoff=0.75)),
        ("RCA cutoff = 1.25",         dict(cutoff=1.25)),
        ("RCA cutoff = 1.50",         dict(cutoff=1.50)),
        ("RCA cutoff = 2.00",         dict(cutoff=2.00)),
        ("Pub threshold = 1",         dict(thr=1)),
        ("Pub threshold = 2",         dict(thr=2)),
        ("Pub threshold = 3",         dict(thr=3)),
        ("Pub threshold = 10",        dict(thr=10)),
        ("Counting: full (not frac)", dict(counting="full")),
        ("Threshold: per-technology", dict(scope="pertech")),
        ("Aggregation: arithmetic",   dict(agg="arith")),
        ("Aggregation: distinct-only",dict(agg="distinct")),
        ("Min output = 10",           dict(min_output=10)),
    ]
    rows = []
    for label, kw in VARIANTS:
        for w in WIN:
            r = rank_variant(long[long.period == w], Xw_all, **kw)
            rho, tau, _ = compare(BASE[w], r, 10)
            _, _, j10   = compare(BASE[w], r, 10)
            _, _, j20   = compare(BASE[w], r, 20)
            rows.append(dict(variant=label, window=w,
                             spearman=round(rho, 3), kendall=round(tau, 3),
                             jac_top10=round(j10, 3), jac_top20=round(j20, 3)))
    S = pd.DataFrame(rows)
    S.to_csv(f"{OUTDIR}/sensitivity_indepth_long.csv", index=False)

    piv = S.pivot_table(index="variant", columns="window", values="spearman")
    piv["mean_rho"] = piv.mean(axis=1)
    piv = piv.reindex([v[0] for v in VARIANTS])
    piv.to_csv(f"{OUTDIR}/sensitivity_indepth_spearman.csv")
    print("\n=== In-depth sensitivity: Spearman vs baseline ===")
    print(piv.round(3).to_string())
    print("\nOverall mean Spearman across variants/windows:",
          round(S.spearman.mean(), 3),
          "| min:", round(S.spearman.min(), 3),
          "at", S.loc[S.spearman.idxmin(), ["variant", "window"]].to_dict())
    return S, piv


# =============================================================================
# 7.  FIGURES
# =============================================================================
def make_figures(all_rank, grid_by_period, indepth_piv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                         "figure.dpi": 130})

    # (A) top-15 composite per window
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (per, tbl) in zip(axes.ravel(), all_rank.items()):
        top = tbl.head(15).iloc[::-1]
        ax.barh(top.index, top["composite"], color="#3b6fb0")
        ax.set_title(per)
        ax.set_xlabel("Composite score (0-1; higher = stronger RCA profile)")
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("Top-15 countries by RCA composite score, per period", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_consensus_top15_periods.png", bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/fig_consensus_top15_periods.pdf", bbox_inches="tight")
    plt.close(fig)

    # (B) sensitivity Spearman surface (cut-off x threshold) per window
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (per, grid) in zip(axes.ravel(), grid_by_period.items()):
        piv = grid.pivot(index="rca_cutoff", columns="pub_threshold",
                         values="spearman")
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis",
                       vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
        ax.set_xlabel("publication threshold"); ax.set_ylabel("RCA cut-off")
        ax.set_title(per)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Ranking stability (Spearman vs baseline): "
                 "RCA-cutoff x publication-threshold grid", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_sensitivity_grid.png", bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/fig_sensitivity_grid.pdf", bbox_inches="tight")
    plt.close(fig)

    # (C) 14-variant in-depth sensitivity heatmap
    piv = indepth_piv.drop(columns=["mean_rho"])
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn",
                   vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    ax.set_title("14-variant robustness: Spearman rho vs baseline\n"
                 "(each variant changes ONE discretionary choice)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_sensitivity_indepth.png", bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/fig_sensitivity_indepth.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8.  MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RCA-based country ranking and sensitivity analysis for Agriculture 4.0."
    )
    parser.add_argument("--input-csv", required=True,
                        help="Classified publication CSV containing Year, countries, and tech_domains.")
    parser.add_argument("--output-dir", default="outputs/rca_sensitivity",
                        help="Directory for CSV tables (default: outputs/rca_sensitivity).")
    parser.add_argument("--figure-dir", default=None,
                        help="Directory for figures (default: <output-dir>/figures).")
    args = parser.parse_args()

    global INFILE, OUTDIR, FIGDIR
    INFILE = str(Path(args.input_csv).expanduser().resolve())
    OUTDIR = str(Path(args.output_dir).expanduser().resolve())
    FIGDIR = str(Path(args.figure_dir).expanduser().resolve()) if args.figure_dir else str(Path(OUTDIR) / "figures")
    if not Path(INFILE).is_file():
        parser.error(f"Input CSV not found: {INFILE}")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)

    print(f">> input: {INFILE}")
    print(f">> output tables: {OUTDIR}")
    print(f">> figures: {FIGDIR}")
    print(">> loading & exploding ...")
    long = load_long(INFILE)
    long.to_csv(f"{OUTDIR}/long_exploded.csv", index=False)
    print(f"   long table: {len(long):,} (paper,country,domain) rows; "
          f"{long['country'].nunique()} countries; "
          f"{long['pid'].nunique():,} papers")

    all_rank, grid_by_period = {}, {}

    Xf_all, Xw_all = build_matrices(long)
    if THRESHOLD_SCOPE == "pooled":
        TAU = tech_thresholds(Xw_all)
        print("Stationary per-tech thresholds (pooled):",
              {k.split()[0]: v for k, v in TAU.items()})
    else:
        TAU = None

    segments = {"ALL 2010-2025": long}
    for a, b in PERIODS:
        segments[f"{a}-{b}"] = long[long["period"] == f"{a}-{b}"]

    for name, seg in segments.items():
        if seg.empty:
            continue
        X_frac, X_whole = build_matrices(seg)
        rca  = compute_rca(X_frac)
        tau  = TAU if TAU is not None else tech_thresholds(X_whole)
        tbl  = rank_countries(seg, X_frac, X_whole, rca, RCA_CUTOFF_BASE, tau)

        rca.to_csv(f"{OUTDIR}/rca_matrix_{name.replace(' ', '_')}.csv")
        tbl.to_csv(f"{OUTDIR}/ranking_{name.replace(' ', '_')}.csv")
        all_rank[name] = tbl

        print(f"\n== {name} ==  ({len(tbl)} countries ranked)")
        print("   top 10 (composite):", ", ".join(tbl.head(10).index.tolist()))
        print("   top 10 (volume)   :",
              ", ".join(tbl.sort_values('papers_whole', ascending=False)
                        .head(10).index.tolist()))

        if name != "ALL 2010-2025":
            cut_df, thr_df, grid_df = sensitivity(seg, X_frac, X_whole, rca,
                                                  base_tau=tau)
            rank_changes = save_country_rank_changes(seg,X_frac,X_whole,rca,name,base_tau=tau)
            cut_df.to_csv(f"{OUTDIR}/sensitivity_cutoff_{name}.csv", index=False)
            thr_df.to_csv(f"{OUTDIR}/sensitivity_threshold_{name}.csv", index=False)
            grid_df.to_csv(f"{OUTDIR}/sensitivity_grid_{name}.csv", index=False)
            grid_by_period[name] = grid_df

    print("\n>> in-depth 14-variant sensitivity ...")
    _, indepth_piv = indepth_sensitivity(long)

    print("\n>> figures ...")
    period_rank = {k: v for k, v in all_rank.items() if k != "ALL 2010-2025"}
    make_figures(period_rank, grid_by_period, indepth_piv)

    print(f"\n>> done. Tables written to {OUTDIR}; figures written to {FIGDIR}.")
    return all_rank


if __name__ == "__main__":
    main()
