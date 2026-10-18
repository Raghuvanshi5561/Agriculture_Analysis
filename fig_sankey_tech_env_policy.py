# -*- coding: utf-8 -*-
"""
fig_sankey_tech_env_policy_updated.py

Three-stage Sankey:
    Technology domain -> Environmental outcome -> Policy priority

CURRENT CLASSIFICATION FILE
---------------------------
Input:
    three_layer_classified_FINAL_v4.csv

Current classification columns:
    tech_domains
    environmental_outcomes
    policy_priorities

No legacy ENV_* or POL_* columns are used.
No tech_primary column is used.

COUNTING
--------
All three classification layers are multi-label.

For a publication with:
    T technology domains
    E environmental outcomes
    P policy priorities

Stage 1:
    technology -> environmental outcome
    weight = 1 / (T * E)

Stage 2:
    environmental outcome -> policy priority
    weight = 1 / (E * P)

Therefore each publication contributes a total weight of exactly 1
to each stage.

OUTPUT
------
Everything is written to:

    outputs/sankey_updated/

Outputs:
    fig_sankey_tech_env_policy.pdf
    fig_sankey_tech_env_policy.png
    sankey_flows.csv
    sankey_node_totals.csv
    sankey_classification_summary.csv

LAYOUT
------
The figure uses a compact layout to reduce unnecessary white space
on the left, right, top and bottom.

COLOUR DESIGN
-------------
A brighter publication-grade palette is used:

    Technology:
        vivid blue

    Environmental outcomes:
        emerald green

    Policy priorities:
        warm orange

The ribbons use lighter versions of the same stage colours.
"""

import builtins
import os
import re

import matplotlib
import numpy as np
import pandas as pd


# =============================================================================
# PROTECT AGAINST SPYDER KERNEL POLLUTION
# =============================================================================

bsum = builtins.sum
bmax = builtins.max
bmin = builtins.min

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402


# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILENAME = "three_layer_classified_FINAL_v4.csv"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", DATA_FILENAME)

FIRST_YEAR = 2010
LATEST_YEAR = 2025

# Separate output folder under outputs/.
OUTPUT_SUBDIR = "sankey_updated"


# =============================================================================
# FIGURE SIZE
# =============================================================================

MM = 1 / 25.4

FIG_W = 190 * MM
FIG_H = 132 * MM

DPI = 600


# =============================================================================
# FLOW DISPLAY
# =============================================================================

# Ribbons below this value are not drawn.
# They remain included in the CSV tables and node totals.
MIN_FLOW = 2.0

SHOW_COUNTS = False


# =============================================================================
# TIGHT SANKEY GEOMETRY
# =============================================================================

NODE_W = 2.2

# Vertical distance between adjacent nodes.
NODE_GAP = 1.65

# Use most of the vertical plotting area.
Y_TOP = 94.0
Y_BOT = 4.5

# Horizontal positions of the three stages.
X = {
    "tech": 14.0,
    "env": 50.0,
    "pol": 86.0,
}

# Tight canvas limits.
XMIN = 5.0
XMAX = 95.0

YMIN = 1.5
YMAX = 98.0


# =============================================================================
# FONT SIZES
# =============================================================================

F_NODE = 7.2
F_COL = 8.2


# =============================================================================
# BRIGHT PUBLICATION-GRADE COLOUR PALETTE
# =============================================================================
#
# These are deliberately brighter than the previous muted palette.
#
# Technology:
#     strong blue
#
# Environmental:
#     vivid emerald green
#
# Policy:
#     warm orange
#
# The "dark" colour is used for nodes and column headings.
# The "mid" colour is used for ribbons.
#
# The colours are intentionally distinct rather than different shades of
# the same colour family, so the three conceptual layers remain immediately
# distinguishable.
# =============================================================================

COL = {

    "tech": {
        "dark": "#1F77B4",     # vivid publication blue
        "mid":  "#5AA6D6",     # lighter blue for ribbons
    },

    "env": {
        "dark": "#1B9E77",     # vivid emerald / teal-green
        "mid":  "#63BE9C",     # lighter green for ribbons
    },

    "pol": {
        "dark": "#E67E22",     # bright warm orange
        "mid":  "#F0AA63",     # lighter orange for ribbons
    },
}


# =============================================================================
# FONT SETTINGS
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# CURRENT SHORT LABELS
# =============================================================================

TECH_LABEL = {

    "D1":
        "AI / ML",

    "D2":
        "Remote sensing",

    "D3":
        "IoT & sensors",

    "D4":
        "Cloud / FMIS",

    "D5":
        "Robotics",

    "D6":
        "Decision support",

    "D7":
        "Blockchain",
}


ENV_LABEL = {

    "E1":
        "GHG mitigation\n& carbon",

    "E2":
        "Soil, land\n& biodiversity",

    "E3":
        "Nutrients &\nagrochemicals",

    "E4":
        "Water use\n& quality",

    "E5":
        "Energy use\nin agriculture",
}


POLICY_LABEL = {

    "P1":
        "Productivity &\nfood security",

    "P2":
        "Climate adaptation\n& mitigation",

    "P3":
        "Climate resilience\n& risk management",

    "P4":
        "Sustainability",

    "P5":
        "Resource-use\nefficiency",
}


# =============================================================================
# CURRENT CATEGORY ORDER
# =============================================================================

TECH_ORDER = [
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
]

ENV_ORDER = [
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
]

POLICY_ORDER = [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
]


# =============================================================================
# PORTABLE PATHS
# =============================================================================

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Input CSV not found: {DATA_PATH}. Place it in the repository data/ folder; see README.md."
    )

DATA = os.path.abspath(DATA_PATH)
OUTDIR = os.path.join(REPO_ROOT, "outputs", OUTPUT_SUBDIR)
os.makedirs(OUTDIR, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def split_labels(cell):
    """
    Split one semicolon-delimited classification cell.
    """

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


def category_code(label, prefix):
    """
    Extract D1/E1/P1-style code.
    """

    m = re.match(
        rf"^({prefix}\d+)\b",
        str(label).strip()
    )

    return (
        m.group(1)
        if m
        else None
    )


def extract_codes(
    cell,
    prefix,
    allowed
):
    """
    Return unique allowed category codes
    from one classification cell.
    """

    out = []
    seen = set()

    for label in split_labels(cell):

        code = category_code(
            label,
            prefix
        )

        if (
            code in allowed
            and code not in seen
        ):

            out.append(code)

            seen.add(code)

    return out


# =============================================================================
# LOAD CURRENT THREE-LAYER FILE
# =============================================================================

print("=" * 78)

print(
    "THREE-STAGE SANKEY: "
    "TECHNOLOGY -> ENVIRONMENT -> POLICY"
)

print("=" * 78)

print(
    f"INPUT : {DATA}"
)

print(
    f"OUTPUT: {OUTDIR}"
)


df = pd.read_csv(
    DATA,
    low_memory=False,
    encoding="utf-8-sig"
)


# =============================================================================
# VALIDATE REQUIRED COLUMNS
# =============================================================================

required = [

    "Year",

    "tech_domains",

    "environmental_outcomes",

    "policy_priorities",
]


missing = [
    c
    for c in required
    if c not in df.columns
]


if missing:

    raise KeyError(
        "The current Sankey requires these "
        "columns, but they are missing: "
        f"{missing}"
    )


# =============================================================================
# RESTRICT ANALYSIS PERIOD
# =============================================================================

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)


df = df[
    df["Year"].between(
        FIRST_YEAR,
        LATEST_YEAR
    )
].copy()


df = df.reset_index(
    drop=True
)


df["_paper_id"] = np.arange(
    len(df)
)


N_CORPUS = len(df)


print(
    f"Corpus ({FIRST_YEAR}-{LATEST_YEAR}): "
    f"{N_CORPUS:,}"
)


# =============================================================================
# BUILD CURRENT CATEGORY ASSIGNMENTS
# =============================================================================

tech_assign = {}
env_assign = {}
policy_assign = {}


for i, row in df.iterrows():

    tech_assign[i] = extract_codes(

        row["tech_domains"],

        "D",

        set(TECH_ORDER)
    )

    env_assign[i] = extract_codes(

        row["environmental_outcomes"],

        "E",

        set(ENV_ORDER)
    )

    policy_assign[i] = extract_codes(

        row["policy_priorities"],

        "P",

        set(POLICY_ORDER)
    )


n_tech = bsum(
    bool(v)
    for v in tech_assign.values()
)


n_env = bsum(
    bool(v)
    for v in env_assign.values()
)


n_policy = bsum(
    bool(v)
    for v in policy_assign.values()
)


all_three = [

    i

    for i in df.index

    if (
        tech_assign[i]
        and
        env_assign[i]
        and
        policy_assign[i]
    )

]


NB = len(
    all_three
)


# =============================================================================
# CLASSIFICATION COVERAGE
# =============================================================================

print(
    "\nCLASSIFICATION COVERAGE"
)


print(
    f"  technology domain assigned       : "
    f"{n_tech:,} "
    f"({100*n_tech/N_CORPUS:.1f}%)"
)


print(
    f"  environmental outcome assigned   : "
    f"{n_env:,} "
    f"({100*n_env/N_CORPUS:.1f}%)"
)


print(
    f"  policy priority assigned         : "
    f"{n_policy:,} "
    f"({100*n_policy/N_CORPUS:.1f}%)"
)


print(
    f"  all three assigned               : "
    f"{NB:,} "
    f"({100*NB/N_CORPUS:.1f}%)"
)


if NB == 0:

    raise ValueError(
        "No publications have all three "
        "current classification layers."
    )


# =============================================================================
# BUILD FRACTIONALLY WEIGHTED FLOWS
# =============================================================================

f1 = {}

f2 = {}


for i in all_three:

    ts = tech_assign[i]

    es = env_assign[i]

    ps = policy_assign[i]


    # -------------------------------------------------------------------------
    # Technology -> Environment
    # -------------------------------------------------------------------------

    w1 = 1.0 / (
        len(ts)
        *
        len(es)
    )


    for t in ts:

        for e in es:

            f1[(t, e)] = (
                f1.get(
                    (t, e),
                    0.0
                )
                +
                w1
            )


    # -------------------------------------------------------------------------
    # Environment -> Policy
    # -------------------------------------------------------------------------

    w2 = 1.0 / (
        len(es)
        *
        len(ps)
    )


    for e in es:

        for p in ps:

            f2[(e, p)] = (
                f2.get(
                    (e, p),
                    0.0
                )
                +
                w2
            )


# =============================================================================
# CONSERVATION CHECKS
# =============================================================================

assert abs(
    bsum(f1.values())
    -
    NB
) < 1e-8, \
    "Stage 1 does not conserve publication weight."


assert abs(
    bsum(f2.values())
    -
    NB
) < 1e-8, \
    "Stage 2 does not conserve publication weight."


print(
    f"\nBoth Sankey stages conserve "
    f"total weight = {NB:,}."
)


# =============================================================================
# NODE TOTALS
# =============================================================================

def source_totals(flows):

    out = {}

    for (
        a,
        _
    ), v in flows.items():

        out[a] = (
            out.get(
                a,
                0.0
            )
            +
            v
        )

    return out


def target_totals(flows):

    out = {}

    for (
        _,
        b
    ), v in flows.items():

        out[b] = (
            out.get(
                b,
                0.0
            )
            +
            v
        )

    return out


t_tot = source_totals(
    f1
)

e_in = target_totals(
    f1
)

e_out = source_totals(
    f2
)

p_tot = target_totals(
    f2
)


# =============================================================================
# FIXED CATEGORY ORDER
# =============================================================================

order_t = [

    x

    for x in TECH_ORDER

    if t_tot.get(
        x,
        0
    ) > 0

]


order_e = [

    x

    for x in ENV_ORDER

    if e_in.get(
        x,
        0
    ) > 0

]


order_p = [

    x

    for x in POLICY_ORDER

    if p_tot.get(
        x,
        0
    ) > 0

]


# =============================================================================
# MIDDLE-NODE CONSERVATION
# =============================================================================

for e in order_e:

    if not np.isclose(

        e_in.get(
            e,
            0.0
        ),

        e_out.get(
            e,
            0.0
        ),

        atol=1e-8

    ):

        raise AssertionError(

            f"Environmental node {e} "
            "does not conserve weight: "

            f"incoming="
            f"{e_in.get(e, 0.0):.8f}, "

            f"outgoing="
            f"{e_out.get(e, 0.0):.8f}"

        )


# =============================================================================
# PRINT NODE TOTALS
# =============================================================================

print(
    "\nNODE TOTALS"
)


print(
    "Technology"
)


for k in order_t:

    print(
        f"  {k}: "
        f"{t_tot[k]:.2f}"
    )


print(
    "Environmental outcomes"
)


for k in order_e:

    print(
        f"  {k}: "
        f"{e_in[k]:.2f}"
    )


print(
    "Policy priorities"
)


for k in order_p:

    print(
        f"  {k}: "
        f"{p_tot[k]:.2f}"
    )


# =============================================================================
# EXPORT FLOW TABLE
# =============================================================================

flow_rows = []


for (
    a,
    b
), v in f1.items():

    flow_rows.append({

        "stage":
            "technology -> environmental outcome",

        "source_code":
            a,

        "source":
            TECH_LABEL.get(
                a,
                a
            ),

        "target_code":
            b,

        "target":
            ENV_LABEL
            .get(
                b,
                b
            )
            .replace(
                "\n",
                " "
            ),

        "weight":
            round(
                v,
                6
            ),
    })


for (
    a,
    b
), v in f2.items():

    flow_rows.append({

        "stage":
            "environmental outcome -> policy priority",

        "source_code":
            a,

        "source":
            ENV_LABEL
            .get(
                a,
                a
            )
            .replace(
                "\n",
                " "
            ),

        "target_code":
            b,

        "target":
            POLICY_LABEL
            .get(
                b,
                b
            )
            .replace(
                "\n",
                " "
            ),

        "weight":
            round(
                v,
                6
            ),
    })


flow_df = pd.DataFrame(
    flow_rows
)


flow_df = flow_df.sort_values(

    [
        "stage",
        "weight",
        "source_code",
        "target_code",
    ],

    ascending=[

        True,
        False,
        True,
        True,
    ]
)


flow_df.to_csv(

    os.path.join(
        OUTDIR,
        "sankey_flows.csv"
    ),

    index=False
)


# =============================================================================
# EXPORT NODE TOTALS
# =============================================================================

node_rows = []


for k in order_t:

    node_rows.append({

        "stage":
            "technology",

        "code":
            k,

        "label":
            TECH_LABEL[k]
            .replace(
                "\n",
                " "
            ),

        "incoming_weight":
            np.nan,

        "outgoing_weight":
            round(
                t_tot[k],
                6
            ),
    })


for k in order_e:

    node_rows.append({

        "stage":
            "environmental outcome",

        "code":
            k,

        "label":
            ENV_LABEL[k]
            .replace(
                "\n",
                " "
            ),

        "incoming_weight":
            round(
                e_in[k],
                6
            ),

        "outgoing_weight":
            round(
                e_out[k],
                6
            ),
    })


for k in order_p:

    node_rows.append({

        "stage":
            "policy priority",

        "code":
            k,

        "label":
            POLICY_LABEL[k]
            .replace(
                "\n",
                " "
            ),

        "incoming_weight":
            round(
                p_tot[k],
                6
            ),

        "outgoing_weight":
            np.nan,
    })


pd.DataFrame(
    node_rows
).to_csv(

    os.path.join(
        OUTDIR,
        "sankey_node_totals.csv"
    ),

    index=False
)


# =============================================================================
# EXPORT CLASSIFICATION SUMMARY
# =============================================================================

summary = pd.DataFrame([

    {
        "metric":
            "corpus_n",

        "value":
            N_CORPUS,
    },

    {
        "metric":
            "technology_assigned_n",

        "value":
            n_tech,
    },

    {
        "metric":
            "environmental_outcome_assigned_n",

        "value":
            n_env,
    },

    {
        "metric":
            "policy_priority_assigned_n",

        "value":
            n_policy,
    },

    {
        "metric":
            "all_three_assigned_n",

        "value":
            NB,
    },

    {
        "metric":
            "all_three_share_pct",

        "value":
            round(
                100 * NB / N_CORPUS,
                3
            ),
    },

    {
        "metric":
            "stage_1_total_weight",

        "value":
            round(
                bsum(
                    f1.values()
                ),
                6
            ),
    },

    {
        "metric":
            "stage_2_total_weight",

        "value":
            round(
                bsum(
                    f2.values()
                ),
                6
            ),
    },

])


summary.to_csv(

    os.path.join(
        OUTDIR,
        "sankey_classification_summary.csv"
    ),

    index=False
)


# =============================================================================
# LAYOUT
# =============================================================================

def stack(
    order,
    totals
):
    """
    Assign vertical node positions while preserving proportional widths.
    """

    span = (

        Y_TOP

        -

        Y_BOT

        -

        NODE_GAP
        *
        (len(order) - 1)

    )


    total = bsum(

        totals[k]

        for k in order

    )


    scale = (
        span
        /
        total
    )


    pos = {}

    y = Y_TOP


    for k in order:

        h = (
            totals[k]
            *
            scale
        )


        pos[k] = (

            y - h,

            y

        )


        y -= (
            h
            +
            NODE_GAP
        )


    return pos, scale


pos_t, sc_t = stack(
    order_t,
    t_tot
)


pos_e, sc_e = stack(
    order_e,
    e_in
)


pos_p, sc_p = stack(
    order_p,
    p_tot
)


# =============================================================================
# DRAW FIGURE
# =============================================================================

fig = plt.figure(

    figsize=(

        FIG_W,
        FIG_H

    ),

    facecolor="white"

)


ax = fig.add_axes(

    [
        0,
        0,
        1,
        1
    ]

)


ax.set_xlim(
    XMIN,
    XMAX
)


ax.set_ylim(
    YMIN,
    YMAX
)


ax.axis("off")


# =============================================================================
# RIBBON
# =============================================================================

def ribbon(

    x0,
    y0a,
    y0b,

    x1,
    y1a,
    y1b,

    colour,
    alpha

):

    """
    Draw one filled cubic ribbon.
    """

    cx = (
        x0
        +
        x1
    ) / 2


    verts = [

        (x0, y0a),

        (cx, y0a),
        (cx, y1a),
        (x1, y1a),

        (x1, y1b),

        (cx, y1b),
        (cx, y0b),
        (x0, y0b),

        (x0, y0a),

    ]


    codes = [

        Path.MOVETO,

        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,

        Path.LINETO,

        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,

        Path.CLOSEPOLY,

    ]


    ax.add_patch(

        PathPatch(

            Path(
                verts,
                codes
            ),

            facecolor=colour,

            alpha=alpha,

            edgecolor="none",

            zorder=2,

        )

    )


# =============================================================================
# DRAW STAGE
# =============================================================================

def draw_stage(

    flows,

    pos_src,
    sc_src,
    x_src,

    pos_tgt,
    sc_tgt,
    x_tgt,

    colour

):

    used_src = {

        k:
        pos_src[k][1]

        for k in pos_src

    }


    used_tgt = {

        k:
        pos_tgt[k][1]

        for k in pos_tgt

    }


    hidden = 0.0


    max_flow = (

        bmax(
            flows.values()
        )

        if flows

        else 1.0

    )


    # Largest ribbons first.
    for (

        (a, b),
        v

    ) in sorted(

        flows.items(),

        key=lambda kv:
            -kv[1]

    ):


        if v < MIN_FLOW:

            hidden += v

            continue


        ha = (
            v
            *
            sc_src
        )


        hb = (
            v
            *
            sc_tgt
        )


        y0b = (
            used_src[a]
            -
            ha
        )


        y0a = used_src[a]


        y1b = (
            used_tgt[b]
            -
            hb
        )


        y1a = used_tgt[b]


        used_src[a] -= ha

        used_tgt[b] -= hb


        alpha = (

            0.24

            +

            0.48
            *
            bmin(
                v / max_flow,
                1.0
            )

        )


        ribbon(

            x_src + NODE_W,

            y0a,
            y0b,

            x_tgt,

            y1a,
            y1b,

            colour,

            alpha,

        )


    return hidden


# =============================================================================
# DRAW BOTH STAGES
# =============================================================================

hid1 = draw_stage(

    f1,

    pos_t,
    sc_t,

    X["tech"],

    pos_e,
    sc_e,

    X["env"],

    COL["tech"]["mid"],

)


hid2 = draw_stage(

    f2,

    pos_e,
    sc_e,

    X["env"],

    pos_p,
    sc_p,

    X["pol"],

    COL["env"]["mid"],

)


print(

    f"\nRibbons below "
    f"MIN_FLOW={MIN_FLOW}: "

    f"stage 1 = "
    f"{hid1:.2f} of {NB:,}; "

    f"stage 2 = "
    f"{hid2:.2f} of {NB:,}."

)


# =============================================================================
# DRAW NODES
# =============================================================================

def draw_nodes(

    order,
    pos,
    totals,
    x,
    key,
    align

):

    labels = {

        "tech":
            TECH_LABEL,

        "env":
            ENV_LABEL,

        "pol":
            POLICY_LABEL,

    }[key]


    for k in order:

        lo, hi = pos[k]


        # ---------------------------------------------------------------------
        # NODE
        # ---------------------------------------------------------------------

        ax.add_patch(

            Rectangle(

                (
                    x,
                    lo
                ),

                NODE_W,

                hi - lo,

                facecolor=
                    COL[key]["dark"],

                edgecolor=
                    "none",

                zorder=5,

            )

        )


        # ---------------------------------------------------------------------
        # LABEL
        # ---------------------------------------------------------------------

        lab = labels.get(
            k,
            k
        )


        if SHOW_COUNTS:

            if "\n" in lab:

                lab = (

                    f"{lab}\n"
                    f"({totals[k]:.1f})"

                )

            else:

                lab = (

                    f"{lab} "
                    f"({totals[k]:.1f})"

                )


        # Middle labels sit over the ribbons.
        bbox = (

            dict(

                boxstyle=
                    "round,pad=0.18",

                facecolor=
                    "white",

                edgecolor=
                    "none",

                alpha=
                    0.90,

            )

            if key == "env"

            else None

        )


        # Keep labels close to nodes to reduce horizontal whitespace.
        if align == "right":

            tx = (
                x
                -
                0.9
            )

        else:

            tx = (
                x
                +
                NODE_W
                +
                0.9
            )


        ax.text(

            tx,

            (
                lo
                +
                hi
            )
            /
            2,

            lab,

            ha=align,

            va="center",

            fontsize=F_NODE,

            color="#222222",

            zorder=6,

            linespacing=1.15,

            bbox=bbox,

        )


draw_nodes(

    order_t,
    pos_t,
    t_tot,

    X["tech"],

    "tech",

    "right"

)


draw_nodes(

    order_e,
    pos_e,
    e_in,

    X["env"],

    "env",

    "right"

)


draw_nodes(

    order_p,
    pos_p,
    p_tot,

    X["pol"],

    "pol",

    "left"

)


# =============================================================================
# COLUMN HEADINGS
# =============================================================================

for key, lab in (

    (
        "tech",
        "TECHNOLOGY DOMAIN"
    ),

    (
        "env",
        "ENVIRONMENTAL OUTCOME"
    ),

    (
        "pol",
        "POLICY PRIORITY"
    ),

):

    ax.text(

        X[key]
        +
        NODE_W / 2,

        Y_TOP
        +
        2.3,

        lab,

        ha="center",

        va="center",

        fontsize=F_COL,

        fontweight="bold",

        color=
            COL[key]["dark"],

    )


# =============================================================================
# SAVE
# =============================================================================

for ext in (

    "pdf",
    "png"

):

    fig.savefig(

        os.path.join(

            OUTDIR,

            f"fig_sankey_tech_env_policy.{ext}"

        ),

        dpi=DPI,

        bbox_inches="tight",

        pad_inches=0.015,

        facecolor="white",

    )


# =============================================================================
# FINAL DIMENSION CHECK
# =============================================================================

fig.canvas.draw()


bb = fig.get_tightbbox(

    fig.canvas.get_renderer()

)


# Correct inches -> millimetres conversion.
w_mm = (
    bb.width
    *
    25.4
)


h_mm = (
    bb.height
    *
    25.4
)


print(

    "\nWROTE "
    "fig_sankey_tech_env_policy "

    f"({w_mm:.1f} x "
    f"{h_mm:.1f} mm)"

)


print(

    "Nominal design size: "

    f"{FIG_W * 25.4:.1f} x "
    f"{FIG_H * 25.4:.1f} mm"

)


if (

    w_mm
    >
    FIG_W * 25.4
    +
    3

):

    print(

        "WARNING: saved width is "

        f"{w_mm - FIG_W * 25.4:.1f} mm wider "

        "than the nominal design because of "
        "label overhang."

    )


# =============================================================================
# OUTPUT SUMMARY
# =============================================================================

print(

    f"\nWROTE "
    f"{os.path.join(OUTDIR, 'sankey_flows.csv')}"

)


print(

    f"WROTE "
    f"{os.path.join(OUTDIR, 'sankey_node_totals.csv')}"

)


print(

    f"WROTE "
    f"{os.path.join(OUTDIR, 'sankey_classification_summary.csv')}"

)


print(
    "\nDONE"
)


plt.close(fig)