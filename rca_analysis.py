# -*- coding: utf-8 -*-
"""
make_rca.py

Revealed Comparative Advantage (RCA) of countries across the seven
Agriculture 4.0 technology domains.

OUTPUT FIGURES
--------------
1. Combined 2 x 2 figure:
       fig_rca_periods.png
       fig_rca_periods.pdf

   Contains:
       (a), (b), (c), (d)

2. Individual figures:
       fig_rca_2010_2013.png/pdf
       fig_rca_2014_2017.png/pdf
       fig_rca_2018_2021.png/pdf
       fig_rca_2022_2025.png/pdf

   Individual figures DO NOT contain (a), (b), (c), (d).

FIGURE FORMAT
-------------
- Countries on Y-axis
- Full country names on Y-axis
- Technology domains on X-axis
- No colourbar / 0-1 scale
- No plot titles
- No period names
- Font size = 11 pt
- Panel labels only in combined figure
- Panel labels outside upper-left corner
- MIN_CELL = 3
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

WRITE = True

MULTI_LABEL = True

MIN_CELL =3

TOP_N = 20

TOP_N_TABLE = 25

PERIODS = [
    (2010, 2013),
    (2014, 2017),
    (2018, 2021),
    (2022, 2025)
]

DPI = 600


# =============================================================================
# FONT SETTINGS
# =============================================================================

mpl.rcParams.update({

    "font.family": "sans-serif",

    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans"
    ],

    "font.size": 16,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "mathtext.default": "regular",
})


# =============================================================================
# INPUT / OUTPUT CONFIGURATION
# =============================================================================
# Values are assigned by main() so importing this module does not run analysis.
INFILE = None
OUTDIR = None
BASE = None
DOMAIN_ORDER = []
DOMAIN_LABEL = {}
DOMAIN_SHORT = {}

# =============================================================================
# COLUMN NAMES
# =============================================================================

COL_YEAR = "Year"

COL_COUNTRIES = "countries"

COL_PRIMARY = "tech_primary"

COL_DOMAINS = "tech_domains"


# =============================================================================
# TECHNOLOGY DOMAIN ORDER  --  READ FROM THE DATA, NOT HARD-CODED
#
# The previous version listed the seven labels as string literals, including
#     "D7 Blockchain & traceability"
# but dictionariesN.py renamed that domain to
#     "D7 Blockchain"
# `long["domain"].isin(DOMAIN_ORDER)` then matched nothing for D7, so 510
# publications were dropped, the D7 column was all zeros, its RCA denominator
# was zero, and the figure rendered an empty column. Nothing raised.
#
# So the labels are read out of the classified file and ordered by D-number,
# exactly as BIB_NEW_PLOTS_frac.py does. Everything downstream is keyed on the
# CODE (D1..D7), so a rename upstream is followed automatically and an eighth
# domain simply appears.
# =============================================================================
def domain_code(label):
    """'D2 Remote sensing & geospatial' -> 'D2'."""
    m = re.match(r"^(D\d+)\b", str(label))
    return m.group(1) if m else str(label)


def resolve_domains(path):
    """Every domain label present in the file, ordered by D-number."""
    col = pd.read_csv(path, low_memory=False, encoding="utf-8-sig",
                      usecols=lambda c: c in (COL_DOMAINS, COL_PRIMARY))
    seen = set()
    for c in (COL_DOMAINS, COL_PRIMARY):
        if c in col.columns:
            for cell in col[c].fillna(""):
                for d in str(cell).split(";"):
                    d = d.strip()
                    if d:
                        seen.add(d)

    def key(lbl):
        m = re.match(r"^D(\d+)$", domain_code(lbl))
        return (0, int(m.group(1))) if m else (1, lbl)

    return sorted(seen, key=key)



# =============================================================================
# TECHNOLOGY DOMAIN LABELS  --  keyed by CODE, so a rename cannot orphan one
# =============================================================================
# CODE_LABEL = {
#     "D1": "AI / ML",
#     "D2": "Remote sensing",
#     "D3": "IoT &\nsensors",
#     "D4": "Cloud / FMIS",
#     "D5": "Robotics",
#     "D6": "Decision\nsupport",
#     "D7": "Blockchain",
# }



CODE_LABEL = {
    "D1": "AI & Machine Learning",
    "D2": "Remote sensing & geospatial",
    "D3": "IoT & sensor systems",
    "D4": "Cloud, data infrastructure & FMIS",
    "D5": "Robotics & autonomous machinery",
    "D6": "Decision support systems",
    "D7": "Blockchain",
}



def short_label(d):
    """Display label; falls back to the first two words of the domain name."""
    c = domain_code(d)
    if c in CODE_LABEL:
        return CODE_LABEL[c]
    return " ".join(re.sub(r"^D\d+\s+", "", str(d)).split()[:2])


DOMAIN_LABEL = {d: short_label(d) for d in DOMAIN_ORDER}
DOMAIN_SHORT = {

    d: d.split(" ", 1)[0]

    for d in DOMAIN_ORDER
}


# =============================================================================
# COUNTRY ISO-2 CODES
# =============================================================================

ISO2 = {

    "Afghanistan": "AF",
    "Albania": "AL",
    "Algeria": "DZ",
    "Argentina": "AR",
    "Armenia": "AM",
    "Australia": "AU",
    "Austria": "AT",
    "Azerbaijan": "AZ",

    "Bahrain": "BH",
    "Bangladesh": "BD",
    "Belarus": "BY",
    "Belgium": "BE",
    "Benin": "BJ",
    "Bolivia": "BO",
    "Bosnia and Herzegovina": "BA",
    "Botswana": "BW",
    "Brazil": "BR",
    "Brunei": "BN",
    "Bulgaria": "BG",
    "Burkina Faso": "BF",

    "Cambodia": "KH",
    "Cameroon": "CM",
    "Canada": "CA",
    "Chile": "CL",
    "China": "CN",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Cuba": "CU",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Czechia": "CZ",

    "Denmark": "DK",
    "Dominican Republic": "DO",

    "Ecuador": "EC",
    "Egypt": "EG",
    "El Salvador": "SV",
    "Estonia": "EE",
    "Ethiopia": "ET",

    "Fiji": "FJ",
    "Finland": "FI",
    "France": "FR",

    "Georgia": "GE",
    "Germany": "DE",
    "Ghana": "GH",
    "Greece": "GR",
    "Guatemala": "GT",
    "Honduras": "HN",
    "Hong Kong": "HK",
    "Hungary": "HU",

    "Iceland": "IS",
    "India": "IN",
    "Indonesia": "ID",
    "Iran": "IR",
    "Iraq": "IQ",
    "Ireland": "IE",
    "Israel": "IL",
    "Italy": "IT",
    "Ivory Coast": "CI",
    "Cote d'Ivoire": "CI",

    "Jamaica": "JM",
    "Japan": "JP",
    "Jordan": "JO",

    "Kazakhstan": "KZ",
    "Kenya": "KE",
    "Kuwait": "KW",
    "Kyrgyzstan": "KG",

    "Laos": "LA",
    "Latvia": "LV",
    "Lebanon": "LB",
    "Libya": "LY",
    "Lithuania": "LT",
    "Luxembourg": "LU",

    "Macedonia": "MK",
    "Madagascar": "MG",
    "Malawi": "MW",
    "Malaysia": "MY",
    "Mali": "ML",
    "Malta": "MT",
    "Mauritius": "MU",
    "Mexico": "MX",
    "Moldova": "MD",
    "Mongolia": "MN",
    "Montenegro": "ME",
    "Morocco": "MA",
    "Mozambique": "MZ",
    "Myanmar": "MM",

    "Namibia": "NA",
    "Nepal": "NP",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nicaragua": "NI",
    "Niger": "NE",
    "Nigeria": "NG",
    "North Korea": "KP",
    "Norway": "NO",

    "Oman": "OM",

    "Pakistan": "PK",
    "Palestine": "PS",
    "Panama": "PA",
    "Papua New Guinea": "PG",
    "Paraguay": "PY",
    "Peru": "PE",
    "Philippines": "PH",
    "Poland": "PL",
    "Portugal": "PT",
    "Puerto Rico": "PR",

    "Qatar": "QA",

    "Romania": "RO",
    "Russia": "RU",
    "Rwanda": "RW",

    "Saudi Arabia": "SA",
    "Senegal": "SN",
    "Serbia": "RS",
    "Sierra Leone": "SL",
    "Singapore": "SG",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Somalia": "SO",
    "South Africa": "ZA",
    "South Korea": "KR",
    "South Sudan": "SS",
    "Spain": "ES",
    "Sri Lanka": "LK",
    "Sudan": "SD",
    "Swaziland": "SZ",
    "Eswatini": "SZ",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Syria": "SY",

    "Taiwan": "TW",
    "Tajikistan": "TJ",
    "Tanzania": "TZ",
    "Thailand": "TH",
    "Togo": "TG",
    "Trinidad and Tobago": "TT",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Turkmenistan": "TM",

    "Uganda": "UG",
    "Ukraine": "UA",
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "United States": "US",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",

    "Venezuela": "VE",
    "Vietnam": "VN",

    "Yemen": "YE",

    "Zambia": "ZM",
    "Zimbabwe": "ZW",

    "Angola": "AO",
    "Barbados": "BB",
    "Bhutan": "BT",
    "Burundi": "BI",
    "Cape Verde": "CV",
    "Central African Republic": "CF",
    "Chad": "TD",
    "Comoros": "KM",
    "Congo": "CG",
    "Democratic Republic of the Congo": "CD",
    "Eritrea": "ER",
    "Gabon": "GA",
    "Gambia": "GM",
    "Guinea": "GN",
    "Guyana": "GY",
    "Haiti": "HT",
    "Lesotho": "LS",
    "Liberia": "LR",
    "Macau": "MO",
    "Maldives": "MV",
    "Mauritania": "MR",
}


# =============================================================================
# ISO -> FULL COUNTRY NAME
# =============================================================================

ISO_TO_COUNTRY = {
    iso: country
    for country, iso in ISO2.items()
}


def iso2(name):

    return ISO2.get(
        name,
        name[:3].upper()
    )


def country_name_from_iso(iso):

    return ISO_TO_COUNTRY.get(
        iso,
        iso
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_long():

    df = pd.read_csv(
        INFILE,
        low_memory=False,
        encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # YEAR
    # -------------------------------------------------------------------------

    df[COL_YEAR] = pd.to_numeric(
        df[COL_YEAR],
        errors="coerce"
    )

    df = df[
        df[COL_YEAR].notna()
    ].copy()

    df[COL_YEAR] = df[
        COL_YEAR
    ].astype(int)


    # -------------------------------------------------------------------------
    # COUNTRY COLUMN
    # -------------------------------------------------------------------------

    if (
        COL_COUNTRIES not in df.columns
        and "Countries" in df.columns
    ):

        df[COL_COUNTRIES] = df["Countries"]


    # -------------------------------------------------------------------------
    # TECHNOLOGY SOURCE
    # -------------------------------------------------------------------------

    src = (
        COL_DOMAINS
        if MULTI_LABEL
        else COL_PRIMARY
    )


    df = df[
        df[src].notna()
        &
        (df[src] != "")
        &
        df[COL_COUNTRIES].notna()
        &
        (df[COL_COUNTRIES] != "")
    ].copy()


    # -------------------------------------------------------------------------
    # PAPER ID
    # -------------------------------------------------------------------------

    df["_pid"] = np.arange(
        len(df)
    )


    # -------------------------------------------------------------------------
    # SPLIT
    # -------------------------------------------------------------------------

    df["country"] = (
        df[COL_COUNTRIES]
        .str.split(";")
    )

    df["domain"] = (
        df[src]
        .str.split(";")
    )


    # -------------------------------------------------------------------------
    # EXPLODE
    # -------------------------------------------------------------------------

    long = (
        df
        .explode("country")
        .explode("domain")
    )


    long["country"] = (
        long["country"]
        .str.strip()
    )

    long["domain"] = (
        long["domain"]
        .str.strip()
    )


    # -------------------------------------------------------------------------
    # KEEP VALID
    # -------------------------------------------------------------------------

    long = long[
        (long["country"].str.len() > 2)
        &
        long["domain"].isin(
            DOMAIN_ORDER
        )
    ]


    # -------------------------------------------------------------------------
    # REMOVE DUPLICATES
    # -------------------------------------------------------------------------

    long = long.drop_duplicates(
        subset=[
            "_pid",
            "country",
            "domain"
        ]
    )


    # -------------------------------------------------------------------------
    # ISO
    # -------------------------------------------------------------------------

    long["ISO"] = (
        long["country"]
        .map(iso2)
    )

    # -------------------------------------------------------------------------
    # UNMAPPED COUNTRIES
    #
    # iso2() falls back to name[:3].upper() for anything missing from ISO2, and
    # country_name_from_iso() cannot reverse that -- so an unmapped country is
    # silently relabelled on the y-axis ("United Arab Emirates" -> "UNI").
    # Anything reaching the top TOP_N_TABLE is reported here. The RCA arithmetic
    # is unaffected; only the label is.
    # -------------------------------------------------------------------------
    ranked = long["country"].value_counts()
    missing = [c for c in ranked.head(TOP_N_TABLE).index if c not in ISO2]
    if missing:
        print("WARNING  no ISO2 entry, y-axis label will be truncated:")
        for c in missing:
            print(f"   {c!r}  (n = {ranked[c]:,})  -> shown as {iso2(c)!r}")
        print("   Add these to the ISO2 dictionary.")


    # -------------------------------------------------------------------------
    # DIAGNOSTICS
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"INPUT: {INFILE}"
    )

    print(
        "Labelling:",
        (
            "MULTI-LABEL (tech_domains)"
            if MULTI_LABEL
            else "PRIMARY ONLY"
        )
    )

    print(
        f"{long['_pid'].nunique():,} papers "
        f"-> {len(long):,} "
        "paper x country x domain rows"
    )

    print("=" * 70)

    return long


# =============================================================================
# RCA MATRIX
# =============================================================================

def rca_matrix(
    sub,
    top_n=TOP_N,
    min_cell=MIN_CELL
):

    # .value_counts() is already ordered by output, descending
    top = (
        sub["ISO"]
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )


    ct = (

        sub[
            sub["ISO"].isin(top)
        ]

        .groupby(
            [
                "ISO",
                "domain"
            ]
        )

        .size()

        .unstack(
            fill_value=0
        )

        .reindex(
            # ROW ORDER
            # ---------
            # This was sorted(top) -- alphabetical by ISO CODE -- while the
            # y-axis displays FULL COUNTRY NAMES. The reader therefore saw
            # "Austria, Australia, Brazil, Canada, China" (AT, AU, BR, CA, CN)
            # and read it as a sorting mistake, because the key doing the
            # sorting is not on the figure.
            # Ordered by publication output instead, which is visible in the
            # data and matches the ranking table. For alphabetical-by-NAME,
            # use:  sorted(top, key=country_name_from_iso)
            index=top,

            columns=DOMAIN_ORDER,

            fill_value=0
        )
    )


    # -------------------------------------------------------------------------
    # RCA
    # -------------------------------------------------------------------------

    row = (
        ct.sum(
            axis=1
        )
        .values
        .reshape(-1, 1)
        .astype(float)
    )


    col = (
        ct.sum(
            axis=0
        )
        .values
        .reshape(1, -1)
        .astype(float)
    )


    tot = float(
        ct.values.sum()
    )


    row[row == 0] = 1
    col[col == 0] = 1


    rca = pd.DataFrame(

        (
            ct.values / row
        )
        /
        (
            col / tot
        ),

        index=ct.index,

        columns=ct.columns
    )


    # -------------------------------------------------------------------------
    # BINARY RCA
    # -------------------------------------------------------------------------

    binary = (
        (rca >= 1)
        &
        (ct >= min_cell)
    ).astype(int)


    suppressed = int(
        (
            (rca >= 1)
            &
            (ct < min_cell)
        )
        .values
        .sum()
    )


    return (
        ct,
        rca,
        binary,
        suppressed
    )


# =============================================================================
# DRAW ONE RCA HEATMAP
# =============================================================================

# def draw_rca_panel(
#     ax,
#     sub,
#     panel_label=None
# ):

#     """
#     Draw one RCA heatmap.

#     IMPORTANT:
#     -----------
#     panel_label is OPTIONAL.

#     If panel_label is supplied:
#         (a), (b), etc. are drawn outside the plot.

#     If panel_label is None:
#         NO panel label is drawn.

#     This is what allows the combined figure and individual
#     figures to behave differently.
#     """

#     (
#         ct,
#         rca,
#         binary,
#         supp
#     ) = rca_matrix(
#         sub
#     )


#     # -------------------------------------------------------------------------
#     # HEATMAP
#     # -------------------------------------------------------------------------

#     ax.imshow(

#         binary.values,

#         cmap="Blues",

#         aspect="auto",

#         interpolation="nearest",

#         vmin=0,

#         vmax=1
#     )


#     # -------------------------------------------------------------------------
#     # MINOR GRID
#     # -------------------------------------------------------------------------

#     ax.set_xticks(

#         np.arange(
#             -0.5,
#             len(DOMAIN_ORDER),
#             1
#         ),

#         minor=True
#     )


#     ax.set_yticks(

#         np.arange(
#             -0.5,
#             len(binary.index),
#             1
#         ),

#         minor=True
#     )


#     ax.grid(

#         which="minor",

#         color="white",

#         linewidth=0.9
#     )


#     ax.tick_params(
#         which="minor",
#         length=0
#     )


#     # -------------------------------------------------------------------------
#     # Y-AXIS: FULL COUNTRY NAMES
#     # -------------------------------------------------------------------------

#     ax.set_yticks(
#         range(
#             len(binary.index)
#         )
#     )


#     country_labels = [

#         country_name_from_iso(
#             iso
#         )

#         for iso in binary.index
#     ]


#     ax.set_yticklabels(

#         country_labels,

#         fontsize=11
#     )


#     ax.tick_params(

#         axis="y",

#         length=6,

#         width=0.8,

#         pad=5
#     )


#     # -------------------------------------------------------------------------
#     # X-AXIS: TECHNOLOGY DOMAINS
#     # -------------------------------------------------------------------------

#     ax.set_xticks(
#         range(
#             len(DOMAIN_ORDER)
#         )
#     )


#     ax.set_xticklabels(

#         [
#             DOMAIN_LABEL[d]
#             for d in DOMAIN_ORDER
#         ],

#         rotation=35,

#         ha="right",

#         va="top",

#         fontsize=11
#     )


#     ax.tick_params(

#         axis="x",

#         length=6,

#         width=0.8,

#         pad=5
#     )


#     # -------------------------------------------------------------------------
#     # OPTIONAL PANEL LABEL
#     # -------------------------------------------------------------------------
#     #
#     # THIS IS THE ONLY CODE THAT CREATES (a), (b), (c), (d).
#     #
#     # For the combined figure:
#     #       panel_label = "(a)"
#     #
#     # For individual figures:
#     #       panel_label = None
#     #
#     # -------------------------------------------------------------------------

#     if panel_label is not None:

#         ax.text(

#             -0.18,1.08,

#             panel_label,

#             transform=ax.transAxes,

#             ha="right",

#             va="top",

#             fontsize=11,

#             fontweight="bold",

#             color="black",

#             clip_on=False,

#             zorder=10
#         )


#     # -------------------------------------------------------------------------
#     # NO TITLES
#     # -------------------------------------------------------------------------

#     ax.set_title("")


#     # -------------------------------------------------------------------------
#     # NO AXIS TITLES
#     # -------------------------------------------------------------------------

#     ax.set_xlabel("")

#     ax.set_ylabel("")


#     # -------------------------------------------------------------------------
#     # REMOVE SPINES
#     # -------------------------------------------------------------------------

#     for spine in ax.spines.values():

#         spine.set_visible(False)


#     return (
#         ct,
#         rca,
#         binary,
#         supp
#     )

def draw_rca_panel(
    ax,
    sub,
    panel_label=None,
    show_technology_labels=True
):
    """
    Draw one RCA heatmap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the heatmap is drawn.

    sub : pandas.DataFrame
        Data for the RCA matrix.

    panel_label : str or None
        Optional panel label: "(a)", "(b)", etc.

    show_technology_labels : bool
        If True:
            Show full technology descriptions below D1-D7.

        If False:
            Show only D1-D7.

    Returns
    -------
    ct, rca, binary, supp
    """

    # =====================================================================
    # RCA MATRIX
    # =====================================================================

    (
        ct,
        rca,
        binary,
        supp
    ) = rca_matrix(sub)


    # =====================================================================
    # HEATMAP
    # =====================================================================

    ax.imshow(
        binary.values,
        cmap="Blues",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1
    )


    # =====================================================================
    # MINOR GRID
    # =====================================================================

    ax.set_xticks(
        np.arange(
            -0.5,
            len(DOMAIN_ORDER),
            1
        ),
        minor=True
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(binary.index),
            1
        ),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=0.9
    )

    ax.tick_params(
        which="minor",
        length=0
    )


    # =====================================================================
    # Y-AXIS: COUNTRY NAMES
    # =====================================================================

    ax.set_yticks(
        range(len(binary.index))
    )

    country_labels = [
        country_name_from_iso(iso)
        for iso in binary.index
    ]

    ax.set_yticklabels(
        country_labels,
        fontsize=18
    )

    ax.tick_params(
        axis="y",
        length=6,
        width=0.8,
        pad=5
    )


    # =====================================================================
    # X-AXIS: D1, D2, ..., D7
    # =====================================================================

    ax.set_xticks(
        range(len(DOMAIN_ORDER))
    )
    DIs=[x.split()[0] for x in DOMAIN_ORDER]
    ax.set_xticklabels(
        #DOMAIN_ORDER,
        DIs,
        fontsize=18,
        rotation=0,
        ha="center",
        va="top"
    )

    ax.tick_params(
        axis="x",
        length=6,
        width=0.8,
        pad=6
    )
    ax.tick_params(
        axis="y",
        length=6,
        width=0.8,
        pad=8
    )


    # =====================================================================
    # OPTIONAL FULL TECHNOLOGY LABELS
    # =====================================================================

    # if show_technology_labels:

    #     row1 = (
    #         "D1: AI & Machine Learning     "
    #         "D2: Remote sensing & geospatial     "
    #         "D3: IoT & sensor systems     "
    #         "D4: Cloud, data infrastructure & FMIS"
    #     )

    #     row2 = (
    #         "D5: Robotics & autonomous machinery     "
    #         "D6: Decision support systems     "
    #         "D7: Blockchain"
    #     )


    #     ax.text(
    #         0.5,
    #         -0.20,
    #         row1,
    #         transform=ax.transAxes,
    #         ha="center",
    #         va="top",
    #         fontsize=18,
    #         color="black",
    #         clip_on=False
    #     )


    #     ax.text(
    #         0.5,
    #         -0.29,
    #         row2,
    #         transform=ax.transAxes,
    #         ha="center",
    #         va="top",
    #         fontsize=18,
    #         color="black",
    #         clip_on=False
    #     )


    # =====================================================================
    # OPTIONAL PANEL LABEL
    # =====================================================================

    if panel_label is not None:

        ax.text(
            -0.18,
            1.08,

            panel_label,

            transform=ax.transAxes,

            ha="right",
            va="top",

            fontsize=16,

            fontweight="bold",

            color="black",

            clip_on=False,

            zorder=10
        )


    # =====================================================================
    # REMOVE TITLES
    # =====================================================================

    ax.set_title("")

    ax.set_xlabel("Technology", fontsize=18)
    ax.set_ylabel("Country", fontsize=18)


    # =====================================================================
    # REMOVE SPINES
    # =====================================================================

    for spine in ax.spines.values():
        spine.set_visible(False)


    # =====================================================================
    # RETURN
    # =====================================================================

    return (
        ct,
        rca,
        binary,
        supp
    )
# =============================================================================
# COMBINED 2 x 2 FIGURE
# =============================================================================

# def plot_combined(long):

#     """
#     Create the 2 x 2 combined figure.

#     ONLY THIS FUNCTION assigns:
#         (a)
#         (b)
#         (c)
#         (d)
#     """

#     fig, axes = plt.subplots(

#         2,
#         2,

#         # 13 x 20 in left a large empty band between the two rows;
#         # bbox_inches='tight' trims the OUTER margin, not the gap
#         # between axes, so it survived into the saved file.
#         figsize=(18, 13.5),

#         squeeze=False
#     )


#     axes = axes.flatten()


#     store = {}


#     panel_labels = [
#         "(a)",
#         "(b)",
#         "(c)",
#         "(d)"
#     ]


#     for k, (
#         y0,
#         y1
#     ) in enumerate(PERIODS):

#         ax = axes[k]


#         sub = long[
#             long[COL_YEAR].between(
#                 y0,
#                 y1
#             )
#         ]


#         (
#             ct,
#             rca,
#             binary,
#             supp
#         ) = draw_rca_panel(

#             ax,

#             sub,

#             panel_label=panel_labels[k]
#         )


#         store[
#             f"{y0}-{y1}"
#         ] = (
#             ct,
#             rca,
#             binary
#         )


#         print(

#             f"\n{y0}-{y1}: "
#             f"{sub['_pid'].nunique():,} papers, "
#             f"{binary.shape[0]} countries, "
#             f"{int(binary.values.sum())} specialisations "
#             f"({supp} suppressed by "
#             f"MIN_CELL={MIN_CELL})"
#         )


#     # -------------------------------------------------------------------------
#     # SPACING
#     # -------------------------------------------------------------------------

#     fig.subplots_adjust(

#         left=0.10,

#         right=0.95,

#         top=0.96,

#         bottom=0.10,

#         wspace=0.30,

#         hspace=0.26
#     )


#     return (
#         fig,
#         store
#     )

def plot_combined(long):

    """
    Create the 2 x 2 combined RCA figure.

    Panel labels:
        (a), (b), (c), (d)

    X-axis:
        D1, D2, ..., D7 on every panel.

    Technology descriptions:
        Shown only once below the bottom row
        (panels c and d), in two rows.
    """

    # =====================================================================
    # FIGURE
    # =====================================================================

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(25, 18),

        squeeze=False
    )

    axes = axes.flatten()

    store = {}

    panel_labels = [
        "(a)",
        "(b)",
        "(c)",
        "(d)"
    ]


    # =====================================================================
    # DRAW FOUR PANELS
    # =====================================================================

    for k, (
        y0,
        y1
    ) in enumerate(PERIODS):

        ax = axes[k]

        sub = long[
            long[COL_YEAR].between(
                y0,
                y1
            )
        ]


        (
            ct,
            rca,
            binary,
            supp
        ) = draw_rca_panel(ax,
    sub,
    panel_label=panel_labels[k],
    show_technology_labels=False
        )


        store[
            f"{y0}-{y1}"
        ] = (
            ct,
            rca,
            binary
        )


        print(

            f"\n{y0}-{y1}: "
            f"{sub['_pid'].nunique():,} papers, "
            f"{binary.shape[0]} countries, "
            f"{int(binary.values.sum())} specialisations "
            f"({supp} suppressed by "
            f"MIN_CELL={MIN_CELL})"
        )


    # =====================================================================
    # SPACING
    # =====================================================================

    fig.subplots_adjust(

        left=0.10,

        right=0.95,

        top=0.96,

        # More space below the bottom row
        # for the technology descriptions
        bottom=0.25,

        wspace=0.30,

        hspace=0.26
    )


    # =====================================================================
    # TECHNOLOGY LABELS
    # SHOWN ONLY ONCE BELOW PANELS (c) AND (d)
    # =====================================================================

    # CODE_LABEL = {

    #     "D1": "AI & Machine Learning",

    #     "D2": "Remote sensing & geospatial",

    #     "D3": "IoT & sensor systems",

    #     "D4": "Cloud, data infrastructure & FMIS",

    #     "D5": "Robotics & autonomous machinery",

    #     "D6": "Decision support systems",

    #     "D7": "Blockchain",
    # }


    # ---------------------------------------------------------------------
    # FIRST ROW OF TECHNOLOGY DESCRIPTIONS
    # ---------------------------------------------------------------------

    row1 = (
        "D1: AI & Machine Learning        "
        "D2: Remote sensing & geospatial        "
        "D3: IoT & sensor systems        "
        "D4: Cloud, data infrastructure & FMIS"
    )


    # ---------------------------------------------------------------------
    # SECOND ROW OF TECHNOLOGY DESCRIPTIONS
    # ---------------------------------------------------------------------

    row2 = (
        "D5: Robotics & autonomous machinery        "
        "D6: Decision support systems        "
        "D7: Blockchain"
    )


    # ---------------------------------------------------------------------
    # Add first row below panels (c) and (d)
    # ---------------------------------------------------------------------

    fig.text(

        0.53,

        0.135,

        row1,

        ha="center",

        va="center",

        fontsize=20,

        color="black"
    )


    # ---------------------------------------------------------------------
    # Add second row
    # ---------------------------------------------------------------------

    fig.text(

        0.53,

        0.085,

        row2,

        ha="center",

        va="center",

        fontsize=20,

        color="black"
    )


    # =====================================================================
    # RETURN
    # =====================================================================

    return (
        fig,
        store
    )
# =============================================================================
# INDIVIDUAL FIGURE
# =============================================================================

def plot_individual(
    long,
    y0,
    y1
):

    """
    Create one individual RCA figure.

    IMPORTANT:
        panel_label=None

    Therefore NO (a), (b), (c), or (d) appears.
    """

    fig, ax = plt.subplots(

        figsize=(9.0, 7.0)
    )


    sub = long[
        long[COL_YEAR].between(
            y0,
            y1
        )
    ]


    (
        ct,
        rca,
        binary,
        supp
    ) = draw_rca_panel(

        ax,

        sub,

        panel_label=None
    )


    # -------------------------------------------------------------------------
    # Individual figure spacing
    # -------------------------------------------------------------------------

    fig.subplots_adjust(

        left=0.15,

        right=0.97,

        top=0.96,

        bottom=0.23
    )


    return (
        fig,
        ct,
        rca,
        binary
    )


# =============================================================================
# RANKING TABLES
# =============================================================================

def ranking_tables(long):

    (
        ct,
        rca,
        binary,
        supp
    ) = rca_matrix(

        long,

        top_n=TOP_N_TABLE
    )


    tot = ct.sum(
        axis=1
    )


    tbl = pd.DataFrame({

        "ISO":
            ct.index,

        "Np":
            tot.values,

        "World share %":
            (
                100
                * tot
                / tot.sum()
            )
            .round(2)
            .values
    })


    # -------------------------------------------------------------------------
    # RCA BY DOMAIN
    # -------------------------------------------------------------------------

    for d in DOMAIN_ORDER:

        tbl[
            DOMAIN_SHORT[d]
        ] = (

            rca[d]

            .round(2)

            .values
        )


    # -------------------------------------------------------------------------
    # NUMBER OF RCA SPECIALISATIONS
    # -------------------------------------------------------------------------

    tbl[
        "Domains RCA>=1"
    ] = (

        binary.sum(
            axis=1
        )

        .values
    )


    # -------------------------------------------------------------------------
    # SORT
    # -------------------------------------------------------------------------

    tbl = (

        tbl

        .sort_values(
            "Np",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


    tbl.insert(

        0,

        "Rank",

        np.arange(
            1,
            len(tbl) + 1
        )
    )


    print(
        "\n"
        + "=" * 70
    )


    print(
        f"RANKING TABLE -- "
        f"top {TOP_N_TABLE} countries, full period"
    )


    print(
        f"({supp} cells had RCA>=1 on fewer than "
        f"{MIN_CELL} publications)"
    )


    print(
        tbl.to_string(
            index=False
        )
    )


    # =========================================================================
    # TOP SPECIALISED COUNTRIES PER DOMAIN
    # =========================================================================

    rows = []


    for d in DOMAIN_ORDER:

        ok = (

            rca[d][
                ct[d] >= MIN_CELL
            ]

            .sort_values(
                ascending=False
            )

            .head(5)
        )


        for r, (
            iso,
            value
        ) in enumerate(
            ok.items(),
            1
        ):

            rows.append({

                "Domain":
                    DOMAIN_SHORT[d],

                "Rank":
                    r,

                "ISO":
                    iso,

                "Country":
                    country_name_from_iso(
                        iso
                    ),

                "RCA":
                    round(
                        value,
                        2
                    ),

                "Np in domain":
                    int(
                        ct.loc[
                            iso,
                            d
                        ]
                    ),

                "Np total":
                    int(
                        tot[iso]
                    )
            })


    top_by_domain = pd.DataFrame(
        rows,
        columns=["Domain", "Rank", "ISO", "Country", "RCA", "Np in domain", "Np total"]
    )


    print(
        "\n"
        + "=" * 70
    )


    print(
        "MOST SPECIALISED COUNTRIES "
        "IN EACH DOMAIN"
    )


    for d in DOMAIN_SHORT.values():

        s = top_by_domain[
            top_by_domain["Domain"] == d
        ]


        print(

            f"\n  {d}: "

            +

            "   ".join(

                f"{r.Country} "
                f"{r.RCA:.2f} "
                f"(n={r['Np in domain']})"

                for _, r in s.iterrows()
            )
        )


    return (
        tbl,
        top_by_domain,
        rca,
        ct
    )


# =============================================================================
# LATEX TABLE
# =============================================================================

def to_latex(tbl):

    head = [

        "Rank",

        "Country",

        "$N_p$",

        "Share (\\%)"

    ] + [

        DOMAIN_SHORT[d]
        for d in DOMAIN_ORDER

    ] + [

        "$n$ RCA$\\geq$1"
    ]


    lines = [

        r"\begin{table*}[t]",

        r"\centering",

        r"\caption{Revealed comparative advantage (RCA) of the leading "
        r"countries across the seven Agriculture 4.0 technology domains, "
        r"2010--2025. RCA $>1$ indicates specialisation relative to the "
        r"field average. Bold marks RCA $\geq 1$ on at least "

        + str(MIN_CELL)

        + r" publications.}",

        r"\label{tab:rca}",

        r"\begin{tabular}{rlrr"

        + "r" * len(DOMAIN_ORDER)

        + "r}",

        r"\toprule",

        " & ".join(head)
        + r" \\",

        r"\midrule"
    ]


    for _, r in tbl.iterrows():

        cells = [

            str(
                int(
                    r["Rank"]
                )
            ),

            str(
                country_name_from_iso(
                    r["ISO"]
                )
            ),

            f"{int(r['Np']):,}",

            f"{r['World share %']:.2f}"
        ]


        for d in DOMAIN_ORDER:

            v = r[
                DOMAIN_SHORT[d]
            ]


            cells.append(

                f"\\textbf{{{v:.2f}}}"

                if v >= 1

                else

                f"{v:.2f}"
            )


        cells.append(

            str(
                int(
                    r["Domains RCA>=1"]
                )
            )
        )


        lines.append(

            " & ".join(cells)
            + r" \\"
        )


    lines += [

        r"\bottomrule",

        r"\end{tabular}",

        r"\end{table*}"
    ]


    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate country-level revealed comparative advantage (RCA) across Agriculture 4.0 technology domains."
    )
    parser.add_argument("--input-csv", required=True,
                        help="Classified publication CSV containing Year, countries, and technology-domain columns.")
    parser.add_argument("--output-dir", default="outputs/rca_only",
                        help="Directory for RCA figures and tables (default: outputs/rca_only).")
    parser.add_argument("--dpi", type=int, default=600,
                        help="Raster export resolution (default: 600 dpi).")
    args = parser.parse_args()

    global INFILE, OUTDIR, BASE, DOMAIN_ORDER, DOMAIN_LABEL, DOMAIN_SHORT, DPI
    INFILE = str(Path(args.input_csv).expanduser().resolve())
    if not Path(INFILE).is_file():
        parser.error(f"Input CSV not found: {INFILE}")
    BASE = Path(INFILE).parent
    OUTDIR = Path(args.output_dir).expanduser().resolve()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    DPI = args.dpi
    DOMAIN_ORDER = resolve_domains(INFILE)
    if not DOMAIN_ORDER:
        parser.error(f"No technology domains found in {INFILE}. Check columns {COL_DOMAINS!r} and {COL_PRIMARY!r}.")
    DOMAIN_LABEL = {d: short_label(d) for d in DOMAIN_ORDER}
    DOMAIN_SHORT = {d: d.split(" ", 1)[0] for d in DOMAIN_ORDER}
    print("Technology domains read from the data:")
    for d in DOMAIN_ORDER:
        print(f"   {d}")

    long = load_long()


    # =============================================================================
    # 1. COMBINED 2 x 2 FIGURE
    # =============================================================================

    fig_combined, store = plot_combined(
        long
    )


    # =============================================================================
    # 2. INDIVIDUAL FIGURES
    # =============================================================================

    # Built ONCE and kept. The previous version built all four figures here,
    # then built all four AGAIN inside the WRITE block: eight renders where four
    # were needed, and the first four were never closed.
    individual_results = {}
    individual_figs = {}


    for (
        y0,
        y1
    ) in PERIODS:

        (
            fig_ind,
            ct,
            rca,
            binary
        ) = plot_individual(

            long,

            y0,

            y1
        )


        individual_figs[f"{y0}-{y1}"] = fig_ind
        individual_results[f"{y0}-{y1}"] = (
            ct,
            rca,
            binary
        )


    # =============================================================================
    # RANKING TABLES
    # =============================================================================

    (
        tbl,
        top_by_domain,
        rca_full,
        ct_full
    ) = ranking_tables(
        long
    )


    # =============================================================================
    # SAVE OUTPUTS
    # =============================================================================

    if WRITE:

        os.makedirs(
            OUTDIR,
            exist_ok=True
        )


        # =========================================================================
        # COMBINED FIGURE
        # =========================================================================

        for ext in (
            "png",
            "pdf"
        ):

            p = os.path.join(

                OUTDIR,

                f"fig_rca_periods.{ext}"
            )


            fig_combined.savefig(

                p,

                dpi=DPI,

                bbox_inches="tight",

                facecolor="white"
            )


            print(
                f"\nwrote {p}"
            )


        # =========================================================================
        # INDIVIDUAL FIGURES
        # =========================================================================

        for (
            y0,
            y1
        ) in PERIODS:

            key = f"{y0}-{y1}"
            fig_ind = individual_figs[key]


            for ext in (
                "png",
                "pdf"
            ):

                p = os.path.join(

                    OUTDIR,

                    f"fig_rca_{key.replace('-', '_')}.{ext}"
                )


                fig_ind.savefig(

                    p,

                    dpi=DPI,

                    bbox_inches="tight",

                    facecolor="white"
                )


                print(
                    f"wrote {p}"
                )


            plt.close(
                fig_ind
            )


        # =========================================================================
        # RCA VALUES FOR EACH PERIOD
        # =========================================================================

        for key, (
            ct,
            rca,
            binary
        ) in store.items():

            rca.round(
                3
            ).to_csv(

                os.path.join(

                    OUTDIR,

                    f"rca_values_{key}.csv"
                )
            )


        # =========================================================================
        # MAIN RANKING TABLE
        # =========================================================================

        tbl.to_csv(

            os.path.join(

                OUTDIR,

                "table_rca_country_domain.csv"
            ),

            index=False
        )


        # =========================================================================
        # TOP SPECIALISED COUNTRIES
        # =========================================================================

        top_by_domain.to_csv(

            os.path.join(

                OUTDIR,

                "table_rca_top_by_domain.csv"
            ),

            index=False
        )


        # =========================================================================
        # FULL-PERIOD COUNTS
        # =========================================================================

        ct_full.to_csv(

            os.path.join(

                OUTDIR,

                "rca_counts_full_period.csv"
            )
        )


        # =========================================================================
        # LATEX TABLE
        # =========================================================================

        with open(

            os.path.join(

                OUTDIR,

                "table_rca_country_domain.tex"
            ),

            "w",

            encoding="utf-8"
        ) as fh:

            fh.write(
                to_latex(tbl)
            )


        print(
            "\n"
            + "=" * 70
        )

        print(
            "ALL OUTPUTS WRITTEN TO:"
        )

        print(
            OUTDIR
        )

        print(
            "=" * 70
        )


    # =============================================================================
    # SHOW COMBINED FIGURE
    # =============================================================================

    plt.show()

if __name__ == "__main__":
    main()
