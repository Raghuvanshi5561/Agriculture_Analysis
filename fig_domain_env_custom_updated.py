# -*- coding: utf-8 -*-
"""
fig_domain_env_custom_updated.py

TECHNOLOGY DOMAIN × ENVIRONMENTAL OUTCOME ANALYSIS
==================================================

Uses the CURRENT three-layer classification file.

LAYER A
-------
Technology domains:
    D1 AI & machine learning
    D2 Remote sensing & geospatial
    D3 IoT & sensor systems
    D4 Cloud, data infrastructure & FMIS
    D5 Robotics & autonomous machinery
    D6 Decision support systems
    D7 Blockchain

LAYER C
-------
Environmental outcomes:
    E1 GHG mitigation & carbon
    E2 Soil, land & biodiversity
    E3 Nutrients & agrochemicals
    E4 Water use & quality
    E5 Energy use in agriculture


IMPORTANT CLASSIFICATION RULE
-----------------------------

The current classification file already contains:

    tech_domains
    environmental_outcomes

Both are MULTI-LABEL fields.

Each publication may therefore contain:

    one or more technology domains
    one or more environmental outcomes

The analysis explodes these fields independently.

Example:

    Publication A:
        tech_domains = D1; D2
        environmental_outcomes = E1; E4

This contributes to:

        D1 × E1
        D1 × E4
        D2 × E1
        D2 × E4

No primary technology assignment is used.

`tech_primary` is completely ignored.


FIGURE ORIENTATION
------------------

TRANSPOSE = True

Therefore the figure displays:

    X-axis = Technology domains
    Y-axis = Environmental outcomes

The analysis itself is always constructed as:

    technology domain × environmental outcome

before the display matrix is transposed.

The transposition is ONLY for visual presentation.


LABELS
------

Technology labels are controlled by CODE_SHORT.

Environmental labels are controlled by ENV_LABEL.

The codes D1-D7 and E1-E5 are extracted from the actual
classification labels and then converted to publication-friendly
display labels.


OUTPUTS
-------

Figures:

    fig_domain_env_absolute.pdf
    fig_domain_env_absolute.png

    fig_domain_env_percent.pdf
    fig_domain_env_percent.png

Tables:

    domain_environment_absolute.csv
    domain_environment_percent.csv
    domain_environment_residuals.csv
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
# SETTINGS / PATHS
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

MM = 1 / 25.4

FIG_W = 200 * MM

FONT = 12

DPI = 600

SHOW_TITLE = False

XROT = 32

# True:
#   X = Technology domains
#   Y = Environmental outcomes
#
# False:
#   X = Environmental outcomes
#   Y = Technology domains
TRANSPOSE = True

AXIS_CATEGORIES = "Environmental outcomes"

AXIS_DOMAINS = "Technology domains"

MIN_CELL = 10

SHOW_FOOTNOTE = False

SHOW_COUNTS = False

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
# DISPLAY LABELS
# =============================================================================

# These labels are ONLY for the figure.
#
# They do NOT alter the classification data or calculations.

CODE_SHORT = {

    "D1": "AI / ML",

    "D2": "Remote sensing",

    "D3": "IoT & sensors",

    "D4": "Cloud / FMIS",

    "D5": "Robotics",

    "D6": "Decision support",

    "D7": "Blockchain",
}


ENV_LABEL = {

    "E1": "GHG mitigation & carbon",

    "E2": "Soil, land & biodiversity",

    "E3": "Nutrients & agrochemicals",

    "E4": "Water use & quality",

    "E5": "Energy use in agriculture",
}


# =============================================================================
# CODE EXTRACTION
# =============================================================================

def code(label, prefix):

    match = re.match(
        rf"^({prefix}\d+)",
        str(label)
    )

    if match:

        return match.group(1)

    return str(label)


# =============================================================================
# SPLIT MULTI-LABEL FIELD
# =============================================================================

def split_labels(cell):

    if pd.isna(cell):

        return []

    value = str(cell).strip()

    if value == "":

        return []

    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]


# =============================================================================
# EXPLODE EXACTLY ONE CLASSIFICATION FIELD
# =============================================================================

def explode_column(
    df,
    column,
    out_col
):

    if column not in df.columns:

        raise KeyError(
            f"Required classification column not found: {column}"
        )

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

    # Original dataframe index is used as publication identifier.
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

    # Important:
    # one paper can have multiple categories,
    # but the same paper/category combination
    # must occur only once.
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

    required = [
        "tech_domains",
        "environmental_outcomes",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise KeyError(
            f"Missing required columns: {missing}"
        )

    print(
        f"INPUT: {DATA}"
    )

    print(
        f"Records: {len(df):,}"
    )

    print(
        "\nUsing classification columns:"
    )

    print(
        "  Technology     : tech_domains"
    )

    print(
        "  Environmental   : environmental_outcomes"
    )

    print(
        "  Primary tech    : NOT USED"
    )

    return df


# =============================================================================
# SORT TECHNOLOGY / ENVIRONMENT CODES
# =============================================================================

def sort_codes(
    values,
    prefix
):

    def key(value):

        match = re.match(
            rf"{prefix}(\d+)",
            str(value)
        )

        if match:

            return (
                0,
                int(match.group(1))
            )

        return (
            1,
            str(value)
        )

    return sorted(
        values,
        key=key
    )


# =============================================================================
# LOAD + EXPLODE
# =============================================================================

df = load()


# -----------------------------------------------------------------------------
# Technology domains
# -----------------------------------------------------------------------------

tech = explode_column(
    df,
    "tech_domains",
    "domain"
)


# -----------------------------------------------------------------------------
# Environmental outcomes
# -----------------------------------------------------------------------------

env = explode_column(
    df,
    "environmental_outcomes",
    "outcome"
)


# =============================================================================
# CATEGORY ORDER
# =============================================================================

doms = sort_codes(
    tech["domain"].unique(),
    "D"
)


outcomes = sort_codes(
    env["outcome"].unique(),
    "E"
)


# Keep only the defined E1-E5 categories.
outcomes = [
    outcome
    for outcome in outcomes
    if code(
        outcome,
        "E"
    ) in ENV_LABEL
]


# Keep only defined D1-D7 categories.
doms = [
    domain
    for domain in doms
    if code(
        domain,
        "D"
    ) in CODE_SHORT
]


# =============================================================================
# PRINT CLASSIFICATION COVERAGE
# =============================================================================

print("\n")
print("=" * 80)
print("TECHNOLOGY DOMAINS")
print("=" * 80)

for domain in doms:

    count = (
        tech
        .loc[
            tech["domain"] == domain,
            "_paper_id"
        ]
        .nunique()
    )

    print(
        f"  {domain}: {count:,}"
    )


print("\n")
print("=" * 80)
print("ENVIRONMENTAL OUTCOMES")
print("=" * 80)

for outcome in outcomes:

    count = (
        env
        .loc[
            env["outcome"] == outcome,
            "_paper_id"
        ]
        .nunique()
    )

    print(
        f"  {outcome}: {count:,}"
    )


# =============================================================================
# TECHNOLOGY × ENVIRONMENT JOIN
# =============================================================================

joined = tech[
    [
        "_paper_id",
        "domain"
    ]
].merge(
    env[
        [
            "_paper_id",
            "outcome"
        ]
    ],
    on="_paper_id",
    how="inner"
)


# Remove duplicate paper-domain-outcome combinations.
joined = joined.drop_duplicates(
    [
        "_paper_id",
        "domain",
        "outcome"
    ]
)


joined = joined[
    joined["domain"].isin(doms)
    &
    joined["outcome"].isin(outcomes)
]


# =============================================================================
# ABSOLUTE CROSS-TAB
# =============================================================================

C = pd.crosstab(
    joined["domain"],
    joined["outcome"]
)


C = C.reindex(
    index=doms,
    columns=outcomes,
    fill_value=0
)


C = C.astype(int)


# =============================================================================
# DOMAIN DENOMINATORS
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

print("\n")
print("=" * 80)
print("ABSOLUTE TECHNOLOGY × ENVIRONMENT")
print("=" * 80)

print(
    C.to_string()
)


print("\n")
print("=" * 80)
print("PERCENTAGE OF TECHNOLOGY DOMAIN")
print("=" * 80)

print(
    P.round(2).to_string()
)


# =============================================================================
# SIGNIFICANCE ANALYSIS
# =============================================================================

obs = C.values.astype(float)


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
    small /
    exp.size
)


cochran_ok = (
    tiny == 0
    and
    frac_small <= 0.20
)


print("\n")
print("=" * 80)
print("CHI-SQUARE ASSUMPTION CHECK")
print("=" * 80)

print(
    f"  expected <5: "
    f"{small}/{exp.size} "
    f"({100 * frac_small:.1f}%)"
)

print(
    f"  expected <1: {tiny}"
)


# =============================================================================
# P-VALUE METHOD
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

        sim = np.array(
            [
                rng.multinomial(
                    int(row_total),
                    col_t / N
                )
                for row_total in row_t
            ],
            dtype=float
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore"
        ):

            expected_sim = (
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

            statistic = np.nansum(
                (
                    sim - expected_sim
                ) ** 2
                /
                np.where(
                    expected_sim > 0,
                    expected_sim,
                    np.nan
                )
            )

        if statistic >= chi2:

            hits += 1

    p_used = (
        hits + 1
    ) / (
        B + 1
    )

    method = (
        f"Monte Carlo, "
        f"{B:,} permutations"
    )


# =============================================================================
# EFFECT SIZE
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
            - 1
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
# RAW RESIDUAL P VALUES
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


# =============================================================================
# HOLM CORRECTION
# =============================================================================

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
# EXPORT TABLES
# =============================================================================

C.to_csv(
    os.path.join(
        TAB,
        "domain_environment_absolute.csv"
    )
)


rows = []


for domain in C.index:

    for outcome in C.columns:

        k = int(
            C.at[
                domain,
                outcome
            ]
        )

        pct, lo, hi = wilson(
            k,
            int(
                domain_n[domain]
            )
        )

        rows.append({

            "domain":
                domain,

            "environmental_outcome":
                outcome,

            "k":
                k,

            "domain_total":
                int(
                    domain_n[domain]
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
                    resid.at[
                        domain,
                        outcome
                    ],
                    2
                ),

            "p_holm":
                round(
                    Padj.at[
                        domain,
                        outcome
                    ],
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
        "domain_environment_percent.csv"
    ),
    index=False
)


resid.round(
    2
).to_csv(
    os.path.join(
        TAB,
        "domain_environment_residuals.csv"
    )
)


# =============================================================================
# DISPLAY LABEL FUNCTIONS
# =============================================================================

def technology_display_label(
    value
):

    c = code(
        value,
        "D"
    )

    return CODE_SHORT.get(
        c,
        value
    )


def environmental_display_label(
    value
):

    c = code(
        value,
        "E"
    )

    return ENV_LABEL.get(
        c,
        value
    )


# =============================================================================
# FIGURE FUNCTION
# =============================================================================

def heat(
    mode,
    fname,
    title
):

    # -------------------------------------------------------------------------
    # IMPORTANT
    # -------------------------------------------------------------------------
    #
    # C and P passed into this function are already in the desired display
    # orientation.
    #
    # When TRANSPOSE=True:
    #
    #     rows    = environmental outcomes
    #     columns = technology domains
    #
    # Therefore:
    #
    #     Y axis = ENV_LABEL
    #     X axis = CODE_SHORT
    #
    # -------------------------------------------------------------------------

    M = (
        C
        if mode == "abs"
        else P
    ).astype(float).values


    # -------------------------------------------------------------------------
    # Figure dimensions
    # -------------------------------------------------------------------------

    fig_height = (
        0.62
        *
        len(
            C.index
        )
        +
        3.0
    )


    fig, ax = plt.subplots(
        figsize=(
            FIG_W,
            fig_height
        )
    )


    # -------------------------------------------------------------------------
    # Heatmap
    # -------------------------------------------------------------------------

    if mode == "abs":

        max_value = M.max()

        norm = PowerNorm(
            gamma=0.5,
            vmin=0,
            vmax=max_value
            if max_value > 0
            else 1
        )

        im = ax.imshow(
            M,
            cmap="Blues",
            aspect="auto",
            norm=norm
        )

    else:

        max_value = M.max()

        im = ax.imshow(
            M,
            cmap="Greens",
            aspect="auto",
            vmin=0,
            vmax=max_value
            if max_value > 0
            else 1
        )


    # -------------------------------------------------------------------------
    # Cell annotations
    # -------------------------------------------------------------------------

    hi = M.max()

    for i in range(
        M.shape[0]
    ):

        for j in range(
            M.shape[1]
        ):

            if hi > 0:

                if mode == "abs":

                    frac = np.sqrt(
                        M[i, j] / hi
                    )

                else:

                    frac = (
                        M[i, j]
                        /
                        hi
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


            dag = (
                "†"
                if k < MIN_CELL
                else ""
            )


            if mode == "abs":

                text = (
                    f"{k:,}"
                    f"{star}"
                )

            else:

                text = (
                    f"{M[i, j]:.2f}"
                    f"{star}"
                    f"{dag}"
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


    # =========================================================================
    # X AXIS
    # =========================================================================

    ax.set_xticks(
        range(
            len(
                C.columns
            )
        )
    )


    # When TRANSPOSE=True, C.columns are TECHNOLOGY DOMAINS.
    #
    # Therefore use CODE_SHORT here.
    #
    # This is the key correction compared with the previous version.

    if TRANSPOSE:

        x_labels = [
            technology_display_label(
                value
            )
            for value in C.columns
        ]

    else:

        x_labels = [
            environmental_display_label(
                value
            )
            for value in C.columns
        ]


    ax.set_xticklabels(
        x_labels,
        rotation=XROT,
        ha="right",
        rotation_mode="anchor",
        fontsize=FONT
    )


    # =========================================================================
    # Y AXIS
    # =========================================================================

    ax.set_yticks(
        range(
            len(
                C.index
            )
        )
    )


    # When TRANSPOSE=True, C.index are ENVIRONMENTAL OUTCOMES.
    #
    # Therefore use ENV_LABEL here.
    #
    # This is the other key correction.

    if TRANSPOSE:

        y_labels = [
            environmental_display_label(
                value
            )
            for value in C.index
        ]

    else:

        y_labels = [
            technology_display_label(
                value
            )
            for value in C.index
        ]


    ax.set_yticklabels(
        y_labels,
        fontsize=FONT
    )


    # =========================================================================
    # TICKS
    # =========================================================================

    ax.tick_params(
        length=0,
        pad=6
    )


    # =========================================================================
    # CELL GRID
    # =========================================================================

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
            len(C.index)
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


    # =========================================================================
    # REMOVE SPINES
    # =========================================================================

    for spine in ax.spines.values():

        spine.set_visible(False)


    # =========================================================================
    # COLORBAR
    # =========================================================================

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

        if not ticks:

            ticks = [0]

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


    # =========================================================================
    # TITLE
    # =========================================================================

    if SHOW_TITLE:

        ax.set_title(
            title,
            fontsize=FONT + 0.5,
            fontweight="bold",
            pad=10
        )


    # =========================================================================
    # AXIS TITLES
    # =========================================================================

    if TRANSPOSE:

        # X = Technology domains
        ax.set_xlabel(
            AXIS_DOMAINS,
            fontsize=FONT + 1,
            fontweight="bold",
            labelpad=12
        )

        # Y = Environmental outcomes
        ax.set_ylabel(
            AXIS_CATEGORIES,
            fontsize=FONT + 1,
            fontweight="bold",
            labelpad=12
        )

    else:

        # X = Environmental outcomes
        ax.set_xlabel(
            AXIS_CATEGORIES,
            fontsize=FONT + 1,
            fontweight="bold",
            labelpad=12
        )

        # Y = Technology domains
        ax.set_ylabel(
            AXIS_DOMAINS,
            fontsize=FONT + 1,
            fontweight="bold",
            labelpad=12
        )


    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.tight_layout()


    # =========================================================================
    # SAVE
    # =========================================================================

    for extension in (
        "pdf",
        "png"
    ):

        output_path = os.path.join(
            FIG,
            f"{fname}.{extension}"
        )

        fig.savefig(
            output_path,
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
# DISPLAY TRANSPOSE
# =============================================================================
#
# IMPORTANT:
#
# All statistics and tables above were calculated using:
#
#       technology × environmental outcome
#
# We transpose ONLY the plotting matrices here.
#
# This means:
#
#       C.index    -> Environmental outcomes
#       C.columns  -> Technology domains
#
# when TRANSPOSE=True.
#
# No classification or statistical calculation is changed.
# =============================================================================

if TRANSPOSE:

    C = C.T

    P = P.T

    Padj = Padj.T

    resid = resid.T


# =============================================================================
# FIGURES
# =============================================================================

heat(
    "abs",
    "fig_domain_env_absolute",
    "Environmental outcomes by technology domain — absolute counts"
)


heat(
    "pct",
    "fig_domain_env_percent",
    "Environmental outcomes by technology domain — share of each domain"
)


# =============================================================================
# SMALL-CELL REPORT
# =============================================================================

flagged = int(
    (
        C.values
        <
        MIN_CELL
    ).sum()
)


print(
    f"\n{flagged} of {C.size} cells "
    f"hold fewer than {MIN_CELL} publications "
    f"and are marked †."
)


# =============================================================================
# CAPTION-READY DESCRIPTION
# =============================================================================

print(
    "\nCAPTION-READY DESCRIPTION"
)


print(
    f"""
Environmental outcomes by technology domain. Values represent the number or
percentage of publications classified simultaneously in the displayed
technology domain and environmental-outcome category. Technology domains and
environmental outcomes are multi-label classifications: each publication is
exploded into its individual categories, so a publication carrying multiple
technologies or outcomes contributes to every applicable pairing. No
primary-technology assignment or outcome grouping is used. Asterisks indicate
standardised residuals significant at p < 0.05 after Holm correction. Daggers
indicate cells with fewer than {MIN_CELL} observed publications. Overall
association: chi-square = {chi2:.0f}, df = {dof}, p = {p_used:.3g}
({method}), Cramer's V = {V:.3f}.
""".strip()
)


print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)