# -*- coding: utf-8 -*-
"""
fig_domain_policy_updated.py

TECHNOLOGY-DOMAIN x POLICY-PRIORITY ANALYSIS
=============================================

CURRENT CLASSIFICATION STRUCTURE
---------------------------------
Technology:
    tech_domains

Policy:
    policy_priorities

IMPORTANT
---------
1. tech_primary is NOT used anywhere.
2. tech_domains is exploded individually.
3. policy_priorities is exploded individually.
4. A paper carrying multiple technology domains contributes to each
   applicable technology-policy pairing.
5. A paper carrying multiple policy priorities contributes to each
   applicable technology-policy pairing.
6. No combined technology category is created.
7. No two technology domains are merged.
8. No two policy priorities are merged.
9. The technology labels shown in the figure are SHORT LABELS based
   strictly on the D1-D7 code.

FIGURE TECHNOLOGY LABELS
------------------------
    D1 AI / ML
    D2 Remote sensing
    D3 IoT & sensors
    D4 Cloud / FMIS
    D5 Robotics
    D6 Decision support
    D7 Blockchain

FIGURE POLICY LABELS
--------------------
    P1 Productivity & food security
    P2 Climate adaptation & mitigation
    P3 Climate resilience & risk management
    P4 Sustainability
    P5 Resource-use efficiency

FIGURE DIMENSIONS
-----------------
Matched to the environmental-outcome figure:

    Width  = 200 mm
    Font   = 12 pt
    DPI    = 600

OUTPUTS
-------
Figures:
    fig_domain_policy_absolute.pdf
    fig_domain_policy_absolute.png
    fig_domain_policy_percent.pdf
    fig_domain_policy_percent.png

Tables:
    domain_policy_absolute.csv
    domain_policy_percent.csv
    domain_policy_residuals.csv
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy import stats


# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILENAME = "three_layer_classified_FINAL_v4.csv"

# Repository layout: input data belongs in ../data/ relative to this script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", DATA_FILENAME)


# =============================================================================
# PORTABLE INPUT / OUTPUT PATHS
# =============================================================================

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Input CSV not found: {DATA_PATH}. Place it in the repository data/ folder; see README.md."
    )

DATA = os.path.abspath(DATA_PATH)
FIG = os.path.join(REPO_ROOT, "outputs", "figures")
TAB = os.path.join(REPO_ROOT, "outputs", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values and rejection flags."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((n - np.arange(n)) * ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted, adjusted < 0.05

def wilson(successes, trials, confidence=0.95):
    """Wilson score interval returned as percentage, lower, upper."""
    from scipy.stats import norm
    if trials <= 0:
        return 0.0, 0.0, 0.0
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / trials
    denom = 1 + z*z/trials
    centre = (p + z*z/(2*trials)) / denom
    half = z * np.sqrt((p*(1-p)/trials) + z*z/(4*trials*trials)) / denom
    return 100*p, 100*max(0.0, centre-half), 100*min(1.0, centre+half)


# =============================================================================
# FIGURE SETTINGS
# =============================================================================
#
# SAME DIMENSIONS AS ENVIRONMENTAL-OUTCOME CODE
# =============================================================================

MM = 1 / 25.4

FIG_W = 250 * MM

FONT = 13

DPI = 600

SHOW_TITLE = False

# Short labels allow a slightly smaller rotation while retaining
# the same overall figure geometry.
XROT = 32

TRANSPOSE = True

AXIS_CATEGORIES = "Policy priorities"

AXIS_DOMAINS = "Technology domains"

MIN_CELL = 10

SEED = 42


# =============================================================================
# MATPLOTLIB SETTINGS
# =============================================================================

mpl.rcParams.update({

    "font.family": "sans-serif",

    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans"
    ],

    "font.size": FONT,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "mathtext.default": "regular",
})


# =============================================================================
# SHORT TECHNOLOGY LABELS
# =============================================================================
#
# IMPORTANT:
#
# The plotting code DOES NOT use the full CSV label.
# It first extracts D1-D7 and then maps the code to the short label.
#
# Therefore these labels work whether the CSV contains:
#
#   D1
#   D1 AI & machine learning
#   D1 AI & machine learning; D2 Remote sensing & geospatial
#
# etc.
# =============================================================================

TECH_SHORT = {

    "D1": "AI / ML",

    "D2": "Remote sensing",

    "D3": "IoT & sensors",

    "D4": "Cloud / FMIS",

    "D5": "Robotics",

    "D6": "Decision support",

    "D7": "Blockchain",
}


# =============================================================================
# POLICY LABELS
# =============================================================================

POLICY_LABEL = {

    "P1":
        "Productivity & food security",

    "P2":
        "Climate adaptation & mitigation",

    "P3":
        "Climate resilience & risk management",

    "P4":
        "Sustainability",

    "P5":
        "Resource-use efficiency",
}


# =============================================================================
# EXTRACT TECHNOLOGY CODE
# =============================================================================

def dcode(label):

    """
    Extract D1-D7 from the beginning of the classification label.

    Examples
    --------
    D1 AI & machine learning
        -> D1

    D2 Remote sensing & geospatial
        -> D2

    D7 Blockchain
        -> D7
    """

    label = str(label).strip()

    match = re.match(
        r"^(D\d+)",
        label,
        flags=re.IGNORECASE
    )

    if match:

        return match.group(1).upper()

    return label


# =============================================================================
# EXTRACT POLICY CODE
# =============================================================================

def pcode(label):

    """
    Extract P1-P5 from the beginning of the policy label.
    """

    label = str(label).strip()

    match = re.match(
        r"^(P\d+)",
        label,
        flags=re.IGNORECASE
    )

    if match:

        return match.group(1).upper()

    return label


# =============================================================================
# SHORT TECHNOLOGY LABEL
# =============================================================================

def technology_short_label(label):

    """
    ALWAYS convert a technology classification to the D1-D7
    short display label.

    This is the important correction.

    The previous version attempted to match the entire classification
    string. This version extracts the D-code first, making the
    display independent of the exact text stored in the CSV.
    """

    code = dcode(label)

    return TECH_SHORT.get(
        code,
        code
    )


# =============================================================================
# POLICY DISPLAY LABEL
# =============================================================================

def policy_display_label(label):

    code = pcode(label)

    return POLICY_LABEL.get(
        code,
        code
    )


# =============================================================================
# SPLIT MULTI-LABEL CELL
# =============================================================================

def split_labels(cell):

    if pd.isna(cell):

        return []

    text = str(cell).strip()

    if text == "":

        return []

    return [
        x.strip()
        for x in text.split(";")
        if x.strip()
    ]


# =============================================================================
# EXPLODE ONE CLASSIFICATION COLUMN
# =============================================================================

def explode_column(
    df,
    column,
    out_col
):

    """
    Explode exactly ONE classification column.

    Each original paper receives a _paper_id.

    Duplicate paper-category combinations are removed.

    This ensures that a paper with:

        D1; D2; D3

    becomes:

        paper 1 -> D1
        paper 1 -> D2
        paper 1 -> D3

    and NOT:

        D1; D2; D3

    as a single technology category.
    """

    d = df[
        df[column].notna()
        &
        (
            df[column]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    d["_paper_id"] = d.index

    d[out_col] = d[column].map(
        split_labels
    )

    d = d.explode(
        out_col,
        ignore_index=False
    )

    d[out_col] = (
        d[out_col]
        .astype(str)
        .str.strip()
    )

    d = d[
        d[out_col] != ""
    ]

    d = d.drop_duplicates(
        [
            "_paper_id",
            out_col
        ]
    )

    return d


# =============================================================================
# LOAD DATA
# =============================================================================

def load():

    df = pd.read_csv(
        DATA,
        low_memory=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------------------
    # Year filter
    # ---------------------------------------------------------

    if "Year" in df.columns:

        df["Year"] = pd.to_numeric(
            df["Year"],
            errors="coerce"
        )

        df = df[
            df["Year"].between(
                2010,
                2025
            )
        ].copy()

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    required = [

        "tech_domains",

        "policy_priorities",

    ]

    missing = [

        c
        for c in required
        if c not in df.columns

    ]

    if missing:

        raise KeyError(
            f"Missing required columns: {missing}"
        )

    print(
        f"\nINPUT: {DATA}"
    )

    print(
        f"Records: {len(df):,}"
    )

    return df


# =============================================================================
# SORT CODES
# =============================================================================

def sort_codes(
    values,
    prefix
):

    def key(x):

        match = re.match(
            rf"^{prefix}(\d+)",
            str(x),
            flags=re.IGNORECASE
        )

        if match:

            return (
                0,
                int(
                    match.group(1)
                )
            )

        return (
            1,
            str(x)
        )

    return sorted(
        values,
        key=key
    )


# =============================================================================
# LOAD
# =============================================================================

df = load()


# =============================================================================
# EXPLODE TECHNOLOGY DOMAINS
# =============================================================================

tech = explode_column(
    df,
    "tech_domains",
    "domain"
)


# =============================================================================
# EXPLODE POLICY PRIORITIES
# =============================================================================

pol = explode_column(
    df,
    "policy_priorities",
    "policy"
)


# =============================================================================
# CATEGORY ORDER
# =============================================================================

doms = sort_codes(
    tech["domain"].unique(),
    "D"
)

pols = sort_codes(
    pol["policy"].unique(),
    "P"
)


# =============================================================================
# KEEP ONLY CURRENT CATEGORIES
# =============================================================================

VALID_D = {
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
}

VALID_P = {
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
}


doms = [

    d

    for d in doms

    if dcode(d) in VALID_D

]


pols = [

    p

    for p in pols

    if pcode(p) in VALID_P

]


# =============================================================================
# PRINT LABEL CHECK
# =============================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "TECHNOLOGY LABEL CHECK"
)

print(
    "=" * 80
)

for d in doms:

    print(
        f"{d:<55} -> "
        f"{technology_short_label(d)}"
    )


print(
    "\n"
    + "=" * 80
)

print(
    "POLICY LABEL CHECK"
)

print(
    "=" * 80
)

for p in pols:

    print(
        f"{p:<55} -> "
        f"{policy_display_label(p)}"
    )


# =============================================================================
# TECHNOLOGY × POLICY PAIRS
# =============================================================================
#
# Each paper is identified by _paper_id.
#
# If a paper has:
#
#   D1; D3
#
# and:
#
#   P1; P4
#
# it contributes to:
#
#   D1 × P1
#   D1 × P4
#   D3 × P1
#   D3 × P4
#
# Each pair is counted ONCE.
# =============================================================================

joined = tech[
    [
        "_paper_id",
        "domain",
    ]
].merge(

    pol[
        [
            "_paper_id",
            "policy",
        ]
    ],

    on="_paper_id",

    how="inner"

)


joined = joined.drop_duplicates(
    [
        "_paper_id",
        "domain",
        "policy",
    ]
)


joined = joined[
    joined["domain"].isin(doms)
    &
    joined["policy"].isin(pols)
]


# =============================================================================
# ABSOLUTE TABLE
# =============================================================================

C = pd.crosstab(

    joined["domain"],

    joined["policy"]

).reindex(

    index=doms,

    columns=pols,

    fill_value=0

).astype(int)


# =============================================================================
# DOMAIN DENOMINATORS
# =============================================================================
#
# Denominator = UNIQUE PAPERS carrying that technology domain.
#
# This is NOT the number of technology-policy pairs.
# =============================================================================

domain_n = (

    tech
    .groupby("domain")["_paper_id"]
    .nunique()
    .reindex(doms)
    .fillna(0)
    .astype(int)

)


# =============================================================================
# PERCENTAGE TABLE
# =============================================================================

P = (

    100

    *

    C.div(
        domain_n,
        axis=0
    )

)


# =============================================================================
# PRINT TABLES
# =============================================================================

print(
    "\nTECHNOLOGY DOMAINS"
)

for d in doms:

    print(
        f"  {technology_short_label(d)}: "
        f"{domain_n[d]:,}"
    )


print(
    "\nPOLICY CATEGORIES"
)

for p in pols:

    print(
        f"  {policy_display_label(p)}"
    )


print(
    "\nABSOLUTE"
)

print(
    C.to_string()
)


print(
    "\nPERCENTAGE OF DOMAIN"
)

print(
    P.round(2).to_string()
)


# =============================================================================
# CHI-SQUARE
# =============================================================================

obs = C.values.astype(
    float
)


chi2, p_asym, dof, exp = (

    stats.chi2_contingency(
        obs
    )

)


small = int(
    (exp < 5).sum()
)

tiny = int(
    (exp < 1).sum()
)

frac_small = (
    small
    /
    exp.size
)

cochran_ok = (

    tiny == 0

    and

    frac_small <= 0.20

)


print(
    "\nCHI-SQUARE ASSUMPTION CHECK"
)

print(
    f"  expected <5: "
    f"{small}/{exp.size} "
    f"({100 * frac_small:.1f}%)"
)

print(
    f"  expected <1: "
    f"{tiny}"
)


# =============================================================================
# P-VALUE
# =============================================================================

if cochran_ok:

    p_used = p_asym

    method = "asymptotic"

else:

    rng = np.random.default_rng(
        SEED
    )

    row_t = obs.sum(
        axis=1
    )

    col_t = obs.sum(
        axis=0
    )

    N = obs.sum()

    hits = 0

    B = 4000

    for _ in range(B):

        sim = np.array([

            rng.multinomial(
                int(rt),
                col_t / N
            )

            for rt in row_t

        ], dtype=float)

        with np.errstate(
            divide="ignore",
            invalid="ignore"
        ):

            e = (

                np.outer(
                    sim.sum(axis=1),
                    sim.sum(axis=0)
                )

                /

                max(
                    sim.sum(),
                    1
                )

            )

            s = np.nansum(

                (sim - e) ** 2

                /

                np.where(
                    e > 0,
                    e,
                    np.nan
                )

            )

        hits += (
            s >= chi2
        )

    p_used = (
        hits + 1
    ) / (
        B + 1
    )

    method = (
        f"Monte Carlo, {B:,} permutations"
    )


# =============================================================================
# CRAMER'S V
# =============================================================================

V = np.sqrt(

    chi2

    /

    (

        obs.sum()

        *

        (
            min(
                obs.shape
            )
            -
            1
        )

    )

)


# =============================================================================
# STANDARDISED RESIDUALS
# =============================================================================

resid = pd.DataFrame(

    (
        obs - exp
    )
    /
    np.sqrt(exp),

    index=C.index,

    columns=C.columns

)


# =============================================================================
# HOLM CORRECTION
# =============================================================================

praw = (

    2

    *

    (
        1
        -
        stats.norm.cdf(
            np.abs(
                resid.values.flatten()
            )
        )
    )

)


padj, _ = holm(
    praw
)


Padj = pd.DataFrame(

    padj.reshape(
        resid.shape
    ),

    index=resid.index,

    columns=resid.columns

)


print(

    f"\nchi2={chi2:.1f}, "
    f"df={dof}, "
    f"p={p_used:.3g} "
    f"({method}), "
    f"Cramer's V={V:.3f}"

)


# =============================================================================
# EXPORT ABSOLUTE TABLE
# =============================================================================

C.to_csv(

    os.path.join(
        TAB,
        "domain_policy_absolute.csv"
    )

)


# =============================================================================
# EXPORT PERCENTAGE TABLE
# =============================================================================

rows = []


for d in C.index:

    for p in C.columns:

        k = int(
            C.at[d, p]
        )

        pct, lo, hi = wilson(

            k,

            int(
                domain_n[d]
            )

        )

        rows.append({

            "domain":
                d,

            "domain_short":
                technology_short_label(d),

            "policy":
                p,

            "policy_short":
                policy_display_label(p),

            "k":
                k,

            "domain_total":
                int(
                    domain_n[d]
                ),

            "pct":
                round(
                    pct,
                    3
                ),

            "ci_lo":
                round(
                    lo,
                    3
                ),

            "ci_hi":
                round(
                    hi,
                    3
                ),

            "z":
                round(
                    resid.at[d, p],
                    2
                ),

            "p_holm":
                round(
                    Padj.at[d, p],
                    4
                ),

            "below_min_cell":
                k < MIN_CELL,

        })


pd.DataFrame(
    rows
).to_csv(

    os.path.join(
        TAB,
        "domain_policy_percent.csv"
    ),

    index=False

)


# =============================================================================
# EXPORT RESIDUALS
# =============================================================================

resid.round(
    2
).to_csv(

    os.path.join(
        TAB,
        "domain_policy_residuals.csv"
    )

)


# =============================================================================
# HEATMAP
# =============================================================================

def heat(
    mode,
    fname,
    title
):

    # -------------------------------------------------------------------------
    # Select matrix
    # -------------------------------------------------------------------------

    M = (

        C

        if mode == "abs"

        else P

    ).astype(
        float
    ).values


    # -------------------------------------------------------------------------
    # SAME FIGURE DIMENSIONS AS ENVIRONMENTAL CODE
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(

        figsize=(

            FIG_W,

            0.62 * len(C) + 4.5

        )

    )


    # -------------------------------------------------------------------------
    # COLOUR SCALE
    # -------------------------------------------------------------------------

    if mode == "abs":

        norm = PowerNorm(

            gamma=0.5,

            vmin=0,

            vmax=M.max()

        )

        im = ax.imshow(

            M,

            cmap="Blues",

            aspect="auto",

            norm=norm

        )

    else:

        im = ax.imshow(

            M,

            cmap="Purples",

            aspect="auto",

            vmin=0,

            vmax=M.max()

        )


    # -------------------------------------------------------------------------
    # CELL VALUES
    # -------------------------------------------------------------------------

    hi = M.max()


    for i in range(
        M.shape[0]
    ):

        for j in range(
            M.shape[1]
        ):

            if hi:

                if mode == "abs":

                    frac = np.sqrt(
                        M[i, j] / hi
                    )

                else:

                    frac = (
                        M[i, j] / hi
                    )

            else:

                frac = 0


            text_color = (

                "white"

                if frac > 0.55

                else "#1a1a1a"

            )


            k = int(
                C.values[i, j]
            )


            star = (

                "*"

                if Padj.iat[i, j] < 0.05

                else ""

            )


            dagger = (

                "†"

                if k < MIN_CELL

                else ""

            )


            if mode == "abs":

                text = (
                    f"{k:,}{star}"
                )

            else:

                text = (

                    f"{M[i, j]:.2f}"
                    f"{star}"
                    f"{dagger}"

                )


            ax.text(

                j,

                i,

                text,

                ha="center",

                va="center",

                fontsize=FONT - 1,

                color=text_color

            )


    # =============================================================================
    # X AXIS
    # =============================================================================
    #
    # THIS IS THE CRITICAL FIX.
    #
    # Instead of:
    #
    #     x -> original full classification string
    #
    # we explicitly do:
    #
    #     x -> D-code -> SHORT LABEL
    #
    # =============================================================================

    ax.set_xticks(

        range(
            len(C.columns)
        )

    )


    x_labels = [

        technology_short_label(
            x
        )

        for x in C.columns

    ]


    ax.set_xticklabels(

        x_labels,

        rotation=XROT,

        ha="right",

        rotation_mode="anchor",

        fontsize=FONT

    )


    # =============================================================================
    # Y AXIS — POLICY
    # =============================================================================

    ax.set_yticks(

        range(
            len(C)
        )

    )


    y_labels = [

        policy_display_label(
            x
        )

        for x in C.index

    ]


    ax.set_yticklabels(

        y_labels,

        fontsize=FONT

    )


    # =============================================================================
    # TICK SETTINGS
    # =============================================================================

    ax.tick_params(

        length=0,

        pad=6

    )


    # =============================================================================
    # CELL GRID
    # =============================================================================

    ax.set_xticks(

        np.arange(
            -0.5,
            len(C.columns)
        ),

        minor=True

    )

    ax.set_yticks(

        np.arange(
            -0.5,
            len(C)
        ),

        minor=True

    )


    ax.grid(

        which="minor",

        color="white",

        linewidth=1.6

    )


    ax.tick_params(

        which="minor",

        length=0

    )


    # =============================================================================
    # REMOVE SPINES
    # =============================================================================

    for sp in ax.spines.values():

        sp.set_visible(
            False
        )


    # =============================================================================
    # COLORBAR
    # =============================================================================

    if mode == "abs":

        ticks = [

            t

            for t in (
                0,
                10,
                50,
                100,
                200,
                350,
                550
            )

            if t <= M.max()

        ]


        # Make sure at least one tick exists.
        if not ticks:

            ticks = [
                0
            ]


        cb = fig.colorbar(

            im,

            ax=ax,

            fraction=0.018,

            pad=0.010,

            ticks=ticks

        )


        cb.set_label(

            "Publications "
            "(square-root colour scale)",

            fontsize=FONT - 1.5

        )

    else:

        cb = fig.colorbar(

            im,

            ax=ax,

            fraction=0.018,

            pad=0.010

        )


        cb.set_label(

            "% of the domain's publications",

            fontsize=FONT - 1.5

        )


    cb.ax.tick_params(

        labelsize=FONT - 1.5

    )


    cb.outline.set_visible(
        False
    )


    # =============================================================================
    # OPTIONAL TITLE
    # =============================================================================

    if SHOW_TITLE:

        ax.set_title(

            title,

            fontsize=FONT + 0.5,

            fontweight="bold",

            pad=10

        )


    # =============================================================================
    # AXIS LABELS
    # =============================================================================

    if TRANSPOSE:

        ax.set_xlabel(

            AXIS_DOMAINS,

            fontsize=FONT + 1,

            fontweight="bold",

            labelpad=12

        )


        ax.set_ylabel(

            AXIS_CATEGORIES,

            fontsize=FONT + 1,

            fontweight="bold",

            labelpad=12

        )

    else:

        ax.set_xlabel(

            AXIS_CATEGORIES,

            fontsize=FONT + 1,

            fontweight="bold",

            labelpad=12

        )


        ax.set_ylabel(

            AXIS_DOMAINS,

            fontsize=FONT + 1,

            fontweight="bold",

            labelpad=12

        )


    # =============================================================================
    # LAYOUT
    # =============================================================================

    fig.tight_layout()


    # =============================================================================
    # SAVE
    # =============================================================================

    for extension in (
        "pdf",
        "png"
    ):

        fig.savefig(

            os.path.join(
                FIG,
                f"{fname}.{extension}"
            ),

            dpi=DPI,

            bbox_inches="tight",

            facecolor="white"

        )


    plt.close(
        fig
    )


    print(
        f"  wrote {fname}.pdf / .png"
    )


# =============================================================================
# TRANSPOSE FOR DISPLAY
# =============================================================================
#
# Original:
#
#     rows    = technology
#     columns = policy
#
# After transpose:
#
#     rows    = policy
#     columns = technology
#
# Therefore the displayed figure has:
#
#     X = Technology domains
#     Y = Policy priorities
#
# =============================================================================

if TRANSPOSE:

    C = C.T

    P = P.T

    Padj = Padj.T

    resid = resid.T


# =============================================================================
# FIGURE 1 — ABSOLUTE COUNTS
# =============================================================================

heat(

    "abs",

    "fig_domain_policy_absolute",

    "Policy priorities by technology domain — absolute counts"

)


# =============================================================================
# FIGURE 2 — PERCENTAGE
# =============================================================================

heat(

    "pct",

    "fig_domain_policy_percent",

    "Policy priorities by technology domain — share of each domain"

)


# =============================================================================
# LOW-CELL REPORT
# =============================================================================

flagged = int(

    (
        C.values
        <
        MIN_CELL
    ).sum()

)


print(

    f"\n{flagged} of {C.size} cells contain fewer than "
    f"{MIN_CELL} publications and are marked †."

)


# =============================================================================
# CAPTION
# =============================================================================

print(
    "\nCAPTION-READY DESCRIPTION"
)


print(

    f"""

Policy priorities by technology domain. Values represent the number or
percentage of publications classified simultaneously in the displayed
technology-domain and policy-priority categories. Technology domains and
policy priorities are multi-label classifications: each publication is
exploded into its individual technology and policy categories, so a
publication carrying multiple technologies or priorities contributes to every
applicable pairing. No primary-technology assignment is used. Technology
domains are displayed using the short D1-D7 labels. Asterisks indicate
standardised residuals significant at p < 0.05 after Holm correction.
Daggers indicate cells with fewer than {MIN_CELL} observed publications.
Overall association: chi-square = {chi2:.0f}, df = {dof},
p = {p_used:.3g} ({method}), Cramer's V = {V:.3f}.

""".strip()

)


# =============================================================================
# FINAL REPORT
# =============================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "DONE — TECHNOLOGY × POLICY ANALYSIS"
)

print(
    "=" * 80
)

print(
    f"Input file       : {DATA}"
)

print(
    f"Figure folder    : {FIG}"
)

print(
    f"Table folder     : {TAB}"
)

print(
    f"Figure width     : {FIG_W / MM:.0f} mm"
)

print(
    f"Figure height    : "
    f"{0.62 * len(C) + 3.0:.2f} inches"
)

print(
    f"Font size        : {FONT} pt"
)

print(
    f"DPI              : {DPI}"
)

print(
    "Technology field : tech_domains"
)

print(
    "Policy field     : policy_priorities"
)

print(
    "Primary tech     : NOT USED"
)

print(
    "Technology labels: D1-D7 SHORT LABELS"
)

print(
    "=" * 80
)