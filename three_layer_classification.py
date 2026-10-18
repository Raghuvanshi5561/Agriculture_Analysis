#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portable repository version of the Agriculture 4.0 three-layer classifier.
"""

# -*- coding: utf-8 -*-
"""
three_layer_classified_FINAL_v4.py

AGRICULTURE 4.0 — THREE-LAYER CLASSIFICATION PIPELINE
======================================================

METHODOLOGICAL VERSION
----------------------

LAYER A — TECHNOLOGY DOMAINS
    D1 AI & machine learning
    D2 Remote sensing & geospatial
    D3 IoT & sensor systems
    D4 Cloud, data infrastructure & FMIS
    D5 Robotics & autonomous machinery
    D6 Decision support systems
    D7 Blockchain

    Evidence fields:
        Title
        Author Keywords
        Indexed Keywords

    Abstract is used only as a fallback when title and keywords yield no technology category.

------------------------------------------------------------

LAYER B — POLICY PRIORITIES
    P1 Productivity & food security
    P2 Climate adaptation & mitigation
    P3 Climate resilience & risk management
    P4 Sustainability
    P5 Resource-use efficiency

    Evidence fields:
        Title
        Author Keywords
        Indexed Keywords
        Abstract

    Abstract is included because policy relevance may be expressed
    at a deeper contextual level than the title/keywords.

------------------------------------------------------------

PRODUCTION VIEW
    production_category mirrors P1 Productivity & food security.

LAYER C — ENVIRONMENTAL OUTCOMES
    E1 GHG mitigation & carbon
    E2 Soil, land & biodiversity
    E3 Nutrients & agrochemicals
    E4 Water use & quality
    E5 Energy use in agriculture

    Evidence fields:
        Title
        Author Keywords
        Indexed Keywords
        Abstract

    Abstract is included because environmental outcomes may be
    demonstrated or described only in the abstract.

------------------------------------------------------------

CLASSIFICATION LOGIC
------------------------------------------------------------

There is NO weighted scoring system.

There is NO threshold.

For each category:

    if dictionary expression occurs in ANY permitted field:
        CLASSIFIED = TRUE

    otherwise:
        CLASSIFIED = FALSE

Therefore:

TECHNOLOGY
    Title OR Author Keywords OR Indexed Keywords
    -> classified

POLICY
    Title OR Author Keywords OR Indexed Keywords OR Abstract
    -> classified

ENVIRONMENT
    Title OR Author Keywords OR Indexed Keywords OR Abstract
    -> classified

------------------------------------------------------------

P3
------------------------------------------------------------

P3 is a NORMAL policy dictionary.

There is NO separate:
    C1
    C2
    C3
    hazard-response logic
    special P3 scoring

P3 is treated exactly like P1, P2, P4 and P5.

Its regex dictionary contains contextualized expressions such as:

    climate-resilient agriculture
    climate-smart agriculture
    climate change adaptation
    adaptive capacity
    drought-tolerant cultivar
    heat-tolerant variety
    flood-resilient farming
    agricultural risk management
    crop risk management
    weather-index insurance
    crop insurance
    early warning systems
    coping strategies

Generic expressions such as:

    drought
    heat
    flood
    risk
    resilience
    insurance

are deliberately NOT sufficient on their own.

------------------------------------------------------------

IMPORTANT
------------------------------------------------------------

The dictionaries are intentionally contextualized.

For example:

    "risk"
        -> NOT P3

    "generic risk assessment"
        -> NOT P3

    "agricultural risk management"
        -> P3

Similarly:

    "resilience"
        -> NOT P3

    "network resilience"
        -> NOT P3

    "climate-resilient agriculture"
        -> P3

------------------------------------------------------------

FIELD-LEVEL AUDIT
------------------------------------------------------------

For every category the script also creates field-specific indicators.

Example:

    POLICY_P3_...                  final classification
    POLICY_P3_..._TITLE            title evidence
    POLICY_P3_..._AUTHOR           author keyword evidence
    POLICY_P3_..._INDEXED          indexed keyword evidence
    POLICY_P3_..._ABSTRACT         abstract evidence

This allows the classification to be audited.

------------------------------------------------------------

OUTPUT
------------------------------------------------------------

Outputs are written to the directory selected with --output-dir (or to the
input CSV's directory if --output-dir is omitted):

    three_layer_classified_comprehensive_normalized.csv
    P3_policy_corpus.csv
    classification_audit.csv
    unclassified_papers.csv
    classification_summary.csv
"""


import os
import re
import argparse
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

BASE = os.path.dirname(os.path.abspath(__file__))

INFILE = "three_layer_classified_comprehensive.csv"
SOURCE_FILE = ""

OUT_CLASSIFIED = "three_layer_classified_comprehensive_normalized.csv"
OUT_P3 = "P3_policy_corpus.csv"
OUT_AUDIT = "classification_audit.csv"
OUT_UNCLASSIFIED = "unclassified_papers.csv"
OUT_SUMMARY = "classification_summary.csv"

WRITE = True


# =============================================================================
# COLUMN CANDIDATES
# =============================================================================

TITLE_CANDIDATES = [
    "Title",
    "title",
]

ABSTRACT_CANDIDATES = [
    "Abstract",
    "abstract",
    "Abstract Text",
    "Abstract text",
]

AUTHOR_KEYWORD_CANDIDATES = [
    "Author Keywords",
    "Author keywords",
    "Author Keywords ",
    "Keywords",
    "AuthorKeyword",
    "Author_Keywords",
    "Author Keywords (DE)",
]

INDEXED_KEYWORD_CANDIDATES = [
    "Index Keywords",
    "Indexed Keywords",
    "Index keywords",
    "Indexed keywords",
    "Index Keywords ",
    "Indexed_Keywords",
    "Index_Keywords",
    "Index Keywords (ID)",
]

YEAR_CANDIDATES = [
    "Year",
    "Publication Year",
    "PY",
]


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def find_column(df, candidates, required=False):

    # Exact match
    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    # Case-insensitive match
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in lookup:
            return lookup[key]

    if required:

        raise KeyError(
            "\nRequired column not found.\n"
            "----------------------------------------\n"
            f"Expected one of:\n{candidates}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def detect_columns(df):

    return {

        "title": find_column(
            df,
            TITLE_CANDIDATES,
            required=True
        ),

        "abstract": find_column(
            df,
            ABSTRACT_CANDIDATES,
            required=False
        ),

        "author_keywords": find_column(
            df,
            AUTHOR_KEYWORD_CANDIDATES,
            required=False
        ),

        "indexed_keywords": find_column(
            df,
            INDEXED_KEYWORD_CANDIDATES,
            required=False
        ),

        "year": find_column(
            df,
            YEAR_CANDIDATES,
            required=False
        ),
    }


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text_series(series):

    if series is None:

        return pd.Series(
            "",
            dtype=object
        )

    cleaned = (
        series
        .fillna("")
        .astype(str)
        .replace({"nan": "", "NaN": "", "None": ""})
    )
    # Normalize punctuation variants for matching: decision-support, neural–network,
    # decision_support and dotted acronyms such as A.I. become searchable consistently.
    cleaned = cleaned.str.replace(r"(?<=\w)\.(?=\w)", "", regex=True)
    cleaned = cleaned.str.replace(r"[-‐‑‒–—−_/]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"[^\w\s]+", " ", regex=True)
    cleaned = cleaned.str.replace("_", " ", regex=False)
    return cleaned.str.replace(r"\s+", " ", regex=True).str.strip()


# =============================================================================
# REGEX NORMALISATION
# =============================================================================

def normalise_regex_pattern(pattern):

    """
    Removes formatting whitespace from multiline regex dictionaries.

    Actual spaces in dictionary expressions are explicitly represented
    using \\s+.
    """

    pattern = str(pattern)

    lines = pattern.splitlines()

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        cleaned.append(line)

    pattern = " ".join(cleaned)

    pattern = re.sub(
        r"\s*\|\s*",
        "|",
        pattern
    )

    pattern = re.sub(
        r"\(\s+",
        "(",
        pattern
    )

    pattern = re.sub(
        r"\s+\)",
        ")",
        pattern
    )

    return pattern.strip()


def compile_dictionary(dictionary):

    compiled = {}

    for name, pattern in dictionary.items():

        cleaned = normalise_regex_pattern(
            pattern
        )

        try:

            compiled[name] = re.compile(
                cleaned,
                flags=re.IGNORECASE
            )

        except re.error as exc:

            raise ValueError(
                "\nREGEX COMPILATION ERROR\n"
                "----------------------------------------\n"
                f"Category : {name}\n"
                f"Pattern  : {cleaned}\n"
                f"Error    : {exc}"
            )

    return compiled


# =============================================================================
# LAYER A — TECHNOLOGY DOMAINS
# =============================================================================

TECH = {

    # -------------------------------------------------------------------------
    # D1
    # -------------------------------------------------------------------------

    "D1 AI & machine learning": r"""
        \bartificial\s+intelligence\b
        |\bai\b
        |\bmachine\s+learning\b
        |\bml\b
        |\bdeep\s+learning\b
        |\bneural\s+network\w*
        |\bcnn\b
        |\brnn\b
        |\blstm\b
        |\bconvolutional\s+neural\w*
        |\bautoml\b
        |\bsupport\s+vector\s+machine\w*
        |\bsvm\b
        |\brandom\s+forest\w*
        |\bgradient\s+boost\w*
        |\bxgboost\b
        |\blightgbm\b
        |\bdecision\s+tree\w*
        |\bk[-\s]nearest\s+neighbo\w*
        |\bknn\b
        |\bensemble\s+learning\b
        |\breinforcement\s+learning\b
        |\btransfer\s+learning\b
        |\bfederated\s+learning\b
        |\bexplainable\s+ai\b
        |\bxai\b
        |\bfuzzy\s+logic\b
        |\bcomputer\s+vision\b
        |\bmachine\s+vision\b
        |\bobject\s+detection\b
        |\bimage\s+classification\b
        |\bsemantic\s+segmentation\b
        |\bimage\s+recognition\b
        |\byolo\w*
        |\btransformer\w*
        |\bvision\s+transformer\w*
        |\bvit\b
        |\bnatural\s+language\s+processing\b
        |\bnlp\b
        |\blarge\s+language\s+model\w*
        |\bllm\w*
        |\bgenerative\s+ai\b
        |\bgenerative\s+artificial\s+intelligence\b
        |\bgenai\b
        |\bgpt[-\s]?\w*\b
        |\bchatgpt\b
        |\bbert\b
        |\broberta\b
        |\balbert\b
        |\bxlnet\b
        |\bt5\b
        |\bllama\w*\b
        |\bgemini\b
        |\bclaude\b
        |\bmistral\w*\b
        |\bfalcon\w*\b
        |\bretrieval[-\s]augmented\s+generation\b
        |\bprompt\s+engineering\b
    """,

    # -------------------------------------------------------------------------
    # D2
    # -------------------------------------------------------------------------

    "D2 Remote sensing & geospatial": r"""
        \bremote\s+sensing\b
        |\bsatellite\s+(?:imag\w*|data)\b
        |\bearth\s+observation\b
        |\blandsat\b
        |\bsentinel[-\s]?\d+[a-z]?\b
        |\bsynthetic\s+aperture\s+radar\b
        |\bsar\s+(?:imag|data)\w*
        |\bndvi\b
        |\bevi\b
        |\blai\b
        |\bfapar\b
        |\bsavi\b
        |\bgndvi\b
        |\bndre\b
        |\bvegetation\s+ind\w*
        |\bhyperspectral\w*
        |\bmultispectral\w*
        |\bthermal\s+imag\w*
        |\bgeospatial\b
        |\bgis\b
        |\bgeographic\s+information\s+system\w*
        |\bunmanned\s+aerial\s+vehicl\w*
        |\buavs?\b
        |\buass?\b
        |\bdrone\w*
        |\blidar\b
        |\blaser\s+scann\w*
        |\baerial\s+(?:imagery|image|photograph\w*)
        |\bphotogrammetr\w*
        |\bgnss\b
        |\bgps\b
        |\bdgps\b
    """,

    # -------------------------------------------------------------------------
    # D3
    # -------------------------------------------------------------------------

    "D3 IoT & sensor systems": r"""
        \binternet\s+of\s+things\b
        |\biot\b
        |\biiot\b
        |\baiot\b
        |\bwireless\s+sensor\s+network\w*
        |\bwsn\b
        |\blorawan\b
        |\blora\b
        |\bnb[-\s]?iot\b
        |\blpwan\b
        |\bzigbee\b
        |\bsmart\s+sensor\w*
        |\bcrop\s+sensor\w*
        |\bsensor[-\s]based\b
        |\bsensor\s+(?:system|technology|data)\w*
        |\bsoil\s+moisture\s+sensor\w*
        |\bembedded\s+sensor\w*
        |\bsensor\s+node\w*
        |\bcyber[-\s]physical\s+system\w*
        |\barduino\b
        |\braspberry\s+pi\b
        |\btelemetr\w*
        |\bactuator\w*
    """,

    # -------------------------------------------------------------------------
    # D4
    # -------------------------------------------------------------------------

    "D4 Cloud, data infrastructure & FMIS": r"""
        \bbig\s+data\b
        |\bcloud\s+computing\b
        |\bcloud\s+platform\w*
        |\bedge\s+computing\b
        |\bfog\s+computing\b
        |\bdata\s+platform\w*
        |\bfarm\s+management\s+information\s+system\w*
        |\bfmis\b
        |\bdigital\s+twin\w*
        |\bdata\s+warehous\w*
        |\bdata\s+pipeline\w*
        |\bdata[-\s]processing\s+infrastructure\w*
        |\bdata\s+(?:collection|processing)\s+infrastructure\w*
        |\bdigital\s+platform\w*
    """,

    # -------------------------------------------------------------------------
    # D5
    # -------------------------------------------------------------------------

    "D5 Robotics & autonomous machinery": r"""
        \brobotic\w*
        |\brobot\w*
        |\bagricultural\s+robot\w*
        |\bautonomous\s+(?:vehicl|machin|tractor|navigat|system)\w*
        |\bself[-\s]driving\b
        |\bmobile\s+robot\w*
        |\bfield\s+robot\w*
        |\bautomat(?:ion|ed)\s+(?:harvest|weed|spray|planting|machinery|application|fertili[sz]er|seeding|spreading)\w*
        |\bautomatic\s+section\s+control\b
        |\btractor\s+guidance\s+system\w*
        |\bharvesting\s+robot\w*
        |\bweeding\s+robot\w*
        |\bend[-\s]effector\w*
        |\bswarm\s+robot\w*
    """,

    # -------------------------------------------------------------------------
    # D6
    # -------------------------------------------------------------------------

    "D6 Decision support systems": r"""
        \bdecision[-\s]support(?:[-\s]system\w*)?\b
        |\bdss\b
        |\badvisory\s+system\w*
        |\bdigital\s+advisor\w*
        |\bexpert\s+system\w*
        |\bknowledge[-\s]based\s+system\w*
        |\brule[-\s]based\s+system\w*
        |\brecommendation\s+system\w*
    """,

    # -------------------------------------------------------------------------
    # D7
    # -------------------------------------------------------------------------

    "D7 Blockchain": r"""
        \bblockchain\w*
        |\bdistributed\s+ledger\w*
        |\bdlt\b
        |\bsmart\s+contract\w*
        |\bhyperledger\b
        |\bdigital\s+traceabilit\w*
        |\bblockchain[-\s]based\s+traceabilit\w*
        |\bblockchain[-\s]enabled\s+traceabilit\w*
    """,
}


# =============================================================================
# LAYER B — POLICY PRIORITIES
# =============================================================================

POLICY = {

    # -------------------------------------------------------------------------
    # P1
    # -------------------------------------------------------------------------

    "P1 Productivity & food security": r"""
        \byield\w*
        |\bcrop\s+yield\w*
        |\bproductivit\w*
        |\bcrop\s+output\b
        |\bharvest\s+efficienc\w*
        |\bcrop\s+performance\b
        |\bfood\s+(?:production|availabilit|suppl|securit)\w*
        |\bfood\s+security\b
        |\bfood\s+self[-\s]?sufficienc\w*
        |\bnutrition\s+securit\w*
        |\bproduction\s+efficienc\w*
        |\bcrop\s+product\w*
        |\bagricultural\s+productivit\w*
        |\bfarm\s+productivit\w*
    """,

    # -------------------------------------------------------------------------
    # P2
    # -------------------------------------------------------------------------

    "P2 Climate adaptation & mitigation": r"""
        \bclimate[-\s]?adapt\w*
        |\bclimate\s+chang\w*\s+adapt\w*
        |\badapt\w*\s+to\s+(?:a\s+)?(?:changing\s+)?climat\w*
        |\badaptation\s+(?:strateg|measure|practice|option|plan|pathway)\w*
        |\badaptive\s+capacit\w*
        |\bclimate\s+chang\w*\s+vulnerabilit\w*
        |\bclimate\s+vulnerabilit\w*
        |\bgreenhouse\s+gas\w*
        |\bghg\b
        |\bemission\s+(?:reduction|mitigation|abatement|intensit)\w*
        |\bnet[-\s]zero\b
        |\blow[-\s]carbon\b
        |\bcarbon[-\s]efficient\b
        |\bclimate\s+(?:change\s+)?mitigat\w*
        |\bdecarboni[sz]\w*
        |\bcarbon\s+(?:neutral\w*|credit\w*|market\w*|pricing|tax\w*|offset\w*)
    """,

    # -------------------------------------------------------------------------
    # P3
    #
    # NORMAL POLICY DICTIONARY.
    # No special-case logic.
    # -------------------------------------------------------------------------

    "P3 Climate resilience & risk management": r"""
        \bclimate[-\s]?resilien\w*
        |\bclimate[-\s]?smart\b
        |\bclimate[-\s]?adapt\w*
        |\bclimate\s+chang\w*\s+adapt\w*
        |\badapt\w*\s+to\s+(?:a\s+)?(?:changing\s+)?climat\w*
        |\badaptation\s+(?:strateg|measure|practice|option|plan|pathway)\w*
        |\badaptive\s+capacit\w*
        |\bclimate\s+risk\s+(?:management|assessment|reduction|mitigation)\b
        |\bagricultur\w*\s+risk\s+(?:management|assessment|reduction|mitigation)\b
        |\bagri[-\s]?food\s+risk\s+(?:management|assessment|reduction|mitigation)\b
        |\bfarm\w*\s+risk\s+(?:management|assessment|reduction|mitigation)\b
        |\bcrop\w*\s+risk\s+(?:management|assessment|reduction|mitigation)\b
        |\bweather\s+risk\s+(?:management|assessment|reduction|mitigation)\b
        |\bdrought\s+risk\s+(?:management|assessment|reduction|mitigation)\b
        |\bflood\s+risk\s+(?:management|assessment|reduction|mitigation)\b
        |\bextreme\s+weather\s+risk\b
        |\bweather[-\s]index\s+insurance\b
        |\bindex[-\s]based\s+insurance\b
        |\bcrop\s+insurance\b
        |\bagricultural\s+insurance\b
        |\bagriculture\s+insurance\b
        |\bfarm\w*\s+insurance\b
        |\bweather\s+insurance\b
        |\bclimate\s+insurance\b
        |\binsurance\s+(?:scheme|product|uptake|penetration)\w*
        |\bearly\s+warning\s+(?:system|service)\w*
        |\bcoping\s+(?:strateg|capacit|mechanism)\w*
        |\bresilien\w*\s+(?:agricultur\w*|farm\w*|crop\w*|food[-\s]?system\w*|agri[-\s]?food\w*)
        |\b(?:agricultur\w*|farm\w*|crop\w*|food[-\s]?system\w*|agri[-\s]?food\w*)[-\s]resilien\w*
        |\b(?:drought|heat|cold|frost|salin\w*|flood|water)[-\s]toleran\w*
        |\b(?:drought|heat|cold|frost|salin\w*|flood|water)[-\s]resistan\w*
        |\btolerant\s+(?:variet|cultivar|genotype|germplasm|line|trait|rootstock)\w*
        |\bresistan\w*\s+(?:variet|cultivar|genotype|crop|strain)\w*
        |\bstress[-\s]resistan\w*
        |\bclimate\s+variabilit\w*
        |\bclimatic\s+variabilit\w*
        |\bclimate\s+chang\w*\s+impact\w*
        |\bimpact\w*\s+of\s+climat\w*
        |\bclimate\s+impact\w*
        |\bcontingency\s+plan\w*
        |\brisk\s+mitigation\s+(?:strategy|measure|practice|plan)\w*
        |\brisk\s+reduction\s+(?:strategy|measure|practice|plan)\w*
    """,

    # -------------------------------------------------------------------------
    # P4
    # -------------------------------------------------------------------------

    "P4 Sustainability": r"""
        \bsustainable\s+(?:agricultur|farming|intensification|production|development)\w*
        |\bsustainabilit\w*\s+(?:goal|target|assessment|transition|indicator)\w*
        |\bcircular\s+econom\w*
        |\bagroecolog\w*
        |\bgreen\s+(?:deal|transition|growth)\b
        |\bsustainable\s+development\s+goal\w*
        |\bsdgs?\b
        |\bsustainable\s+intensification\b
        |\benvironmental\s+sustainabilit\w*
        |\bsustainable\s+food\s+system\w*
    """,

    # -------------------------------------------------------------------------
    # P5
    # -------------------------------------------------------------------------

    "P5 Resource-use efficiency": r"""
        \bresource\s+use\s+efficienc\w*
        |\binput\s+(?:use\s+)?efficienc\w*
        |\bnutrient\s+(?:use\s+efficienc|recovery)\w*
        |\bfertili[sz]er\s+(?:optimi[sz]ation|efficienc)\w*
        |\bnitrogen\s+use\s+efficienc\w*
        |\bphosphorus\s+use\s+efficienc\w*
        |\bagronomic\s+efficienc\w*
        |\beco[-\s]?efficienc\w*
        |\benergy\s+(?:use\s+)?efficienc\w*
        |\bwater[-\s]use\s+efficienc\w*
        |\bwater\s+productivit\w*
        |\birrigation\s+(?:efficienc|performance)\w*
        |\breduced\s+(?:pesticide|agrochemical|input)\w*
        |\b(?:pesticide|agrochemical)\s+(?:use\s+)?reduction\b
        |\binput\s+optimi[sz]ation\b
        |\breduced\s+input\s+use\b
        |\binput\s+saving\w*
    """,
}


# =============================================================================
# LAYER C — ENVIRONMENTAL OUTCOMES
# =============================================================================

ENVIRONMENT = {

    # -------------------------------------------------------------------------
    # E1
    # -------------------------------------------------------------------------

    "E1 GHG mitigation & carbon": r"""
        \bgreenhouse\s+gas\s+emission\w*
        |\bgreenhouse\s+gas\w*
        |\bghg\s+emission\w*
        |\bghg\s+reduction\w*
        |\bghg\s+mitigation\w*
        |\bcarbon\s+(?:footprint|emission\w*|dioxide|budget|balance|accounting|credit\w*|offset\w*|neutral\w*|intensit\w*|pricing)
        |\bco2\s+(?:emission|flux|footprint|equivalent|reduction|uptake)\w*
        |\bch4\b
        |\bmethane\b
        |\bn2o\b
        |\bnitrous\s+oxide\b
        |\bglobal\s+warming\s+potential\b
        |\bgwp\b
        |\bemission\s+(?:reduction|mitigation|intensit|factor|inventor)\w*
        |\bclimate\s+(?:change\s+)?mitigation\b
        |\bcarbon\s+sequestration\b
        |\bsoil\s+carbon\s+sequestration\b
        |\blow[-\s]carbon\s+(?:agricultur|farming|production)\w*
        |\bdecarboni[sz]\w*
        |\bnet[-\s]zero\s+(?:emission|carbon)\w*
        |\bcarbon\s+stock\w*
        |\bcarbon\s+storage\b
    """,

    # -------------------------------------------------------------------------
    # E2
    # -------------------------------------------------------------------------

    "E2 Soil, land & biodiversity": r"""
        \bsoil\s+organic\s+(?:carbon|matter)\b
        |\bsoil\s+organic\s+carbon\s+sequestration\b
        |\bsoil\s+carbon\b
        |\bsoil\s+health\b
        |\bsoil\s+qualit\w*
        |\bsoil\s+fertilit\w*
        |\bsoil\s+microbi\w*
        |\bsoil\s+biodiversit\w*
        |\bsoil\s+soc\b
        |\bsoc\s+stock\w*
        |\bsoc\s+content\b
        |\bcarbon\s+stock\w*
        |\bcarbon\s+farming\b
        |\bsoil\s+respiration\b
        |\bsoil\s+structure\b
        |\bcover\s+crop\w*
        |\bconservation\s+tillage\b
        |\bno[-\s]till\w*
        |\bsoil\s+compaction\b
        |\b(?:soil|water|wind|gully|rill)\s+erosion\b
        |\berosion\s+(?:risk|control)\b
        |\bland\s+degradation\b
        |\bdesertificat\w*
        |\bsoil\s+loss\b
        |\bland\s+use\s+change\b
        |\bland[-\s]use\s+conversion\b
        |\bdeforestation\b
        |\bland\s+degradation\s+neutralit\w*
        |\bbiodiversit\w*
        |\bagrobiodiversit\w*
        |\bspecies\s+richness\b
        |\becosystem\s+service\w*
        |\bpollinator\w*
        |\bpollination\s+service\w*
        |\bhabitat\s+(?:loss|quality|fragmentation|restoration|connectivit)\w*
        |\bbeneficial\s+insect\w*
        |\bnatural\s+enem\w*
        |\bfunctional\s+diversit\w*
        |\blandscape\s+heterogeneit\w*
        |\bagroforestr\w*
        |\bagroecolog\w*
        |\bhedgerow\w*
        |\bwildlife\b
        |\bsoil\s+fauna\b
    """,

    # -------------------------------------------------------------------------
    # E3
    # -------------------------------------------------------------------------

    "E3 Nutrients & agrochemicals": r"""
        \bnutrient\s+(?:use\s+)?efficienc\w*
        |\bnue\b
        |\bnitrogen\s+use\s+efficienc\w*
        |\bphosphorus\s+use\s+efficienc\w*
        |\bfertili[sz]\w+\s+(?:use|application|management|optimi[sz]|reduction|rate|efficienc|saving)\w*
        |\bnutrient\s+management\b
        |\bprecision\s+(?:nutrient|fertili[sz]\w+)\w*
        |\bvariable\s+rate\s+fertili[sz]\w*
        |\b(?:nitrate|nitrogen|phosphorus|phosphate|nutrient)\s+(?:leach|runoff|loss|surplus|pollut)\w*
        |\bnitrogen\s+(?:balance|budget|surplus)\b
        |\bexcess\s+fertili[sz]\w*
        |\bover[-\s]fertili[sz]\w*
        |\bsite[-\s]specific\s+nutrient\s+management\b
        |\b(?:pesticide|herbicide|fungicide|insecticide|agrochemical|chemical)\s*\w*\s*(?:reduction|saving|use\s+reduction|minimi[sz]|lower\w*|efficienc|optimi[sz])\w*
        |\breduc\w+\s+(?:pesticide|herbicide|fungicide|insecticide|agrochemical|chemical\s+input)\w*
        |\b(?:site[-\s]specific|targeted|selective|spot|variable\s+rate|precision)\s+(?:spray|application|weed\s+control|herbicide)\w*
        |\bspot\s+spray\w*
        |\bintegrated\s+pest\s+management\b
        |\bipm\b
        |\bpesticide\s+(?:use|load|application\s+rate)\b
        |\bchemical\s+input\s+reduction\b
        |\bbiological\s+control\b
    """,

    # -------------------------------------------------------------------------
    # E4
    # -------------------------------------------------------------------------

    "E4 Water use & quality": r"""
        \bwater\s+(?:use\s+)?efficienc\w*
        |\bwue\b
        |\bwater\s+(?:productivit|saving|conservation|footprint|withdrawal|abstraction|consumption|scarcit|demand\s+management)\w*
        |\birrigation\s+(?:efficienc|scheduling|optimi[sz]|management|water\s+use)\w*
        |\b(?:deficit|precision|smart|drip|micro)\s+irrigation\b
        |\bgroundwater\s+(?:depletion|extraction|recharge|conservation)\w*
        |\bevapotranspiration\s+(?:based|scheduling|estimation)\w*
        |\bwater\s+resource\s+management\b
        |\bcrop\s+water\s+requirement\w*
        |\beutrophicat\w*
        |\balgal\s+bloom\w*
        |\bwater\s+(?:qualit|pollut|contaminat)\w*
        |\bnon[-\s]point\s+source\s+pollut\w*
        |\bagricultural\s+(?:runoff|pollution|effluent)\b
        |\bnitrate\s+(?:contaminat|concentration|pollut)\w*
        |\bpesticide\s+(?:residue|contaminat|runoff)\w*
        |\bwastewater\s+(?:reuse|treatment|irrigation)\b
        |\bsalini[sz]ation\b
        |\bsalt[-\s]affected\s+soil\w*
    """,

    # -------------------------------------------------------------------------
    # E5
    # -------------------------------------------------------------------------

    "E5 Energy use in agriculture": r"""
        \benergy\s+(?:use|consumption|demand|requirement|intensit|efficienc|saving|saving\s+potential)\w*
        |\benergy[-\s]use\s+efficienc\w*
        |\benergy\s+efficienc\w*
        |\benergy\s+consumption\b
        |\benergy\s+demand\b
        |\benergy\s+requirement\w*
        |\benergy\s+intensit\w*
        |\benergy\s+saving\w*
        |\benergy\s+management\b
        |\bagricultur\w*\s+energy\s+(?:use|consumption|efficienc|demand|management)\w*
        |\bfarm\s+energy\s+(?:use|consumption|efficienc|management)\w*
        |\brenewable\s+energy\s+(?:in|for|on)\s+(?:agricultur|farm|farming)\w*
        |\bsolar\s+energy\s+(?:in|for)\s+(?:agricultur|farm|farming)\w*
        |\bagricultural\s+(?:machiner|tractor|pump|irrigation)\w*\s+(?:energy|fuel)
        |\bfuel\s+consumption\s+(?:in\s+)?agricultur\w*
        |\bdiesel\s+consumption\s+(?:in\s+)?agricultur\w*
    """,
}


# =============================================================================
# COMPILE DICTIONARIES
# =============================================================================

TECH_RX = compile_dictionary(TECH)

POLICY_RX = compile_dictionary(POLICY)

ENV_RX = compile_dictionary(ENVIRONMENT)


# =============================================================================
# TEXT PREPARATION
# =============================================================================

def empty_series(index):

    return pd.Series(
        "",
        index=index,
        dtype=object
    )


def prepare_text(df, columns):

    text = {}

    text["title"] = clean_text_series(
        df[columns["title"]]
    )

    if columns["author_keywords"] is not None:

        text["author_keywords"] = clean_text_series(
            df[
                columns["author_keywords"]
            ]
        )

    else:

        text["author_keywords"] = empty_series(
            df.index
        )

    if columns["indexed_keywords"] is not None:

        text["indexed_keywords"] = clean_text_series(
            df[
                columns["indexed_keywords"]
            ]
        )

    else:

        text["indexed_keywords"] = empty_series(
            df.index
        )

    if columns["abstract"] is not None:

        text["abstract"] = clean_text_series(
            df[
                columns["abstract"]
            ]
        )

    else:

        text["abstract"] = empty_series(
            df.index
        )

    return text


# =============================================================================
# MATCHING
# =============================================================================

def contains(rx, series):

    return series.str.contains(
        rx,
        regex=True,
        na=False
    )


# =============================================================================
# CATEGORY MATCHING
# =============================================================================

def classify_dictionary(
    df,
    patterns,
    text,
    include_abstract=False,
    prefix=""
):

    """
    Direct dictionary matching.

    NO WEIGHTS.
    NO THRESHOLD.

    A category is classified if the dictionary matches in at least
    one permitted field.

    Technology:
        title + author + indexed

    Policy/environment:
        title + author + indexed + abstract
    """

    results = pd.DataFrame(
        index=df.index
    )

    for category, rx in patterns.items():

        # ---------------------------------------------------------
        # Field-level evidence
        # ---------------------------------------------------------

        title_hit = contains(
            rx,
            text["title"]
        )

        author_hit = contains(
            rx,
            text["author_keywords"]
        )

        indexed_hit = contains(
            rx,
            text["indexed_keywords"]
        )

        if include_abstract:

            abstract_hit = contains(
                rx,
                text["abstract"]
            )

        else:

            abstract_hit = pd.Series(
                False,
                index=df.index
            )

        # ---------------------------------------------------------
        # Final classification
        # ---------------------------------------------------------

        final_hit = (
            title_hit
            |
            author_hit
            |
            indexed_hit
            |
            (
                abstract_hit
                if include_abstract
                else False
            )
        )

        # ---------------------------------------------------------
        # Store field-level evidence
        # ---------------------------------------------------------

        safe = safe_name(category)

        results[f"{prefix}_{safe}"] = final_hit.astype(bool)

        results[
            f"{prefix}_{safe}_TITLE"
        ] = title_hit.astype(bool)

        results[
            f"{prefix}_{safe}_AUTHOR"
        ] = author_hit.astype(bool)

        results[
            f"{prefix}_{safe}_INDEXED"
        ] = indexed_hit.astype(bool)

        results[
            f"{prefix}_{safe}_ABSTRACT"
        ] = abstract_hit.astype(bool)

    return results


# =============================================================================
# SAFE COLUMN NAME
# =============================================================================

def safe_name(name):

    return (
        name
        .replace(" ", "_")
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )


# =============================================================================
# EXTRACT FINAL CATEGORY FLAGS
# =============================================================================

def extract_final_flags(
    evidence,
    categories,
    prefix
):

    flags = pd.DataFrame(
        index=evidence.index
    )

    for category in categories:

        safe = safe_name(
            category
        )

        flags[category] = evidence[
            f"{prefix}_{safe}"
        ].astype(bool)

    return flags


# =============================================================================
# BUILD MULTI-LABEL STRING
# =============================================================================

def build_multilabel(flags):

    return flags.apply(
        lambda row: "; ".join(
            [
                category
                for category in flags.columns
                if bool(row[category])
            ]
        ),
        axis=1
    )


# =============================================================================
# PRIMARY TECHNOLOGY
# =============================================================================

def determine_primary_technology(
    flags
):

    """
    Since there is no scoring system, primary technology is simply
    the first matched technology according to dictionary order.

    This is only a convenience field.

    The authoritative field remains tech_domains.
    """

    primary = pd.Series(
        "",
        index=flags.index,
        dtype=object
    )

    for idx in flags.index:

        matched = [
            category
            for category in flags.columns
            if bool(flags.at[idx, category])
        ]

        if matched:

            primary.at[idx] = matched[0]

    return primary


# =============================================================================
# SELF-TEST
# =============================================================================

def regex_match(
    dictionary,
    category,
    phrase
):

    return bool(
        dictionary[category].search(
            phrase
        )
    )


def self_test():

    print("\n")
    print("=" * 80)
    print("RUNNING SELF-TEST")
    print("=" * 80)

    failures = []

    # =========================================================================
    # TECHNOLOGY
    # =========================================================================

    technology_tests = [

        (
            "convolutional neural network for weed detection",
            True
        ),

        (
            "UAV multispectral imagery and NDVI",
            True
        ),

        (
            "LoRaWAN soil moisture sensor node",
            True
        ),

        (
            "blockchain traceability in the food supply chain",
            True
        ),

        (
            "leaf area index (LAI) retrieval",
            True
        ),

        ("Decision-support system for crop management", True),
        ("neural-network crop disease classifier", True),
        ("A.I./M.L. for precision agriculture", True),
    ]

    domain_punctuation_tests = [
        ("Decision-support system", "D6 Decision support systems"),
        ("neural-network classifier", "D1 AI & machine learning"),
        ("sensor-based nitrogen management", "D3 IoT & sensor systems"),
        ("UAVs and LiDAR imagery", "D2 Remote sensing & geospatial"),
        ("data-processing infrastructure", "D4 Cloud, data infrastructure & FMIS"),
        ("automated application system", "D5 Robotics & autonomous machinery"),
    ]
    for phrase, category in domain_punctuation_tests:
        normalized_phrase = clean_text_series(pd.Series([phrase])).iloc[0]
        if not TECH_RX[category].search(normalized_phrase):
            failures.append(f"\nTECHNOLOGY PUNCTUATION\nText: {phrase!r}\nExpected category: {category}")

    for phrase, expected in technology_tests:

        normalized_phrase = clean_text_series(pd.Series([phrase])).iloc[0]
        got = any(
            rx.search(normalized_phrase)
            for rx in TECH_RX.values()
        )

        if bool(got) != expected:

            failures.append(
                "\nTECHNOLOGY\n"
                f"Text     : {phrase!r}\n"
                f"Expected : {expected}\n"
                f"Got      : {got}"
            )

    # =========================================================================
    # POLICY
    # =========================================================================

    policy_tests = [

        (
            "agricultural productivity and crop yield",
            True
        ),

        (
            "climate change adaptation strategies",
            True
        ),

        (
            "sustainable agriculture and SDGs",
            True
        ),

        (
            "resource use efficiency in farming",
            True
        ),

        (
            "agricultural risk management under climate change",
            True
        ),

        (
            "weather-index insurance for farmers",
            True
        ),

        (
            "climate-resilient agriculture",
            True
        ),

        (
            "climate-smart agriculture",
            True
        ),

        (
            "climate change adaptation strategies in agriculture",
            True
        ),

        (
            "drought-tolerant wheat cultivar",
            True
        ),

        (
            "heat-tolerant maize variety",
            True
        ),

        (
            "drought adaptation strategies for farmers",
            True
        ),

        (
            "flood-resilient farming systems",
            True
        ),
    ]

    for phrase, expected in policy_tests:

        normalized_phrase = clean_text_series(pd.Series([phrase])).iloc[0]
        got = any(
            rx.search(normalized_phrase)
            for rx in POLICY_RX.values()
        )

        if bool(got) != expected:

            failures.append(
                "\nPOLICY\n"
                f"Text     : {phrase!r}\n"
                f"Expected : {expected}\n"
                f"Got      : {got}"
            )

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================

    environment_tests = [

        (
            "greenhouse gas emissions from paddy fields",
            True
        ),

        (
            "soil organic carbon sequestration",
            True
        ),

        (
            "nitrogen use efficiency in maize",
            True
        ),

        (
            "irrigation water use efficiency",
            True
        ),

        (
            "energy consumption in agricultural machinery",
            True
        ),
    ]

    for phrase, expected in environment_tests:

        normalized_phrase = clean_text_series(pd.Series([phrase])).iloc[0]
        got = any(
            rx.search(normalized_phrase)
            for rx in ENV_RX.values()
        )

        if bool(got) != expected:

            failures.append(
                "\nENVIRONMENT\n"
                f"Text     : {phrase!r}\n"
                f"Expected : {expected}\n"
                f"Got      : {got}"
            )

    # =========================================================================
    # P3 NEGATIVE TESTS
    # =========================================================================

    p3_category = (
        "P3 Climate resilience & risk management"
    )

    p3_negative_tests = [

        "drought stress in wheat",

        "heat stress in maize",

        "network resilience under node failure",

        "organizational resilience in digital systems",

        "flood fill segmentation algorithm",

        "generic insurance analysis",

        "generic risk assessment",

        "risk assessment in computer networks",

        "network risk management",
    ]

    for phrase in p3_negative_tests:

        got = regex_match(
            POLICY_RX,
            p3_category,
            phrase
        )

        if got:

            failures.append(
                "\nP3 NEGATIVE\n"
                f"Text     : {phrase!r}\n"
                "Expected : False\n"
                f"Got      : {got}"
            )

    # =========================================================================
    # FIELD LOGIC TESTS
    # =========================================================================

    test_df = pd.DataFrame({

        "Title": [
            "climate-resilient agriculture"
        ],

        "Author Keywords": [
            ""
        ],

        "Index Keywords": [
            ""
        ],

        "Abstract": [
            ""
        ],
    })

    test_text = {

        "title": clean_text_series(
            test_df["Title"]
        ),

        "author_keywords": clean_text_series(
            test_df["Author Keywords"]
        ),

        "indexed_keywords": clean_text_series(
            test_df["Index Keywords"]
        ),

        "abstract": clean_text_series(
            test_df["Abstract"]
        ),
    }

    evidence = classify_dictionary(
        test_df,
        POLICY_RX,
        test_text,
        include_abstract=True,
        prefix="POLICY"
    )

    p3_safe = safe_name(
        p3_category
    )

    got = bool(
        evidence[
            f"POLICY_{p3_safe}"
        ].iloc[0]
    )

    if not got:

        failures.append(
            "\nFIELD LOGIC\n"
            "P3 in title should classify.\n"
            "Expected : True\n"
            f"Got      : {got}"
        )

    # -------------------------------------------------------------------------
    # Abstract-only test
    # -------------------------------------------------------------------------

    test_df2 = pd.DataFrame({

        "Title": [
            "IoT irrigation system"
        ],

        "Author Keywords": [
            ""
        ],

        "Index Keywords": [
            ""
        ],

        "Abstract": [
            "The system improved irrigation water use efficiency."
        ],
    })

    test_text2 = {

        "title": clean_text_series(
            test_df2["Title"]
        ),

        "author_keywords": clean_text_series(
            test_df2["Author Keywords"]
        ),

        "indexed_keywords": clean_text_series(
            test_df2["Index Keywords"]
        ),

        "abstract": clean_text_series(
            test_df2["Abstract"]
        ),
    }

    env_evidence = classify_dictionary(
        test_df2,
        ENV_RX,
        test_text2,
        include_abstract=True,
        prefix="ENV"
    )

    e4_safe = safe_name(
        "E4 Water use & quality"
    )

    got = bool(
        env_evidence[
            f"ENV_{e4_safe}"
        ].iloc[0]
    )

    if not got:

        failures.append(
            "\nABSTRACT LOGIC\n"
            "E4 should classify from abstract.\n"
            "Expected : True\n"
            f"Got      : {got}"
        )

    # -------------------------------------------------------------------------
    # Technology abstract fallback test
    # -------------------------------------------------------------------------

    test_df3 = pd.DataFrame({

        "Title": [
            "Agricultural productivity"
        ],

        "Author Keywords": [
            ""
        ],

        "Index Keywords": [
            ""
        ],

        "Abstract": [
            "The study applies machine learning."
        ],
    })

    test_text3 = {

        "title": clean_text_series(
            test_df3["Title"]
        ),

        "author_keywords": clean_text_series(
            test_df3["Author Keywords"]
        ),

        "indexed_keywords": clean_text_series(
            test_df3["Index Keywords"]
        ),

        "abstract": clean_text_series(
            test_df3["Abstract"]
        ),
    }

    tech_evidence = classify_dictionary(
        test_df3,
        TECH_RX,
        test_text3,
        include_abstract=False,
        prefix="TECH"
    )

    d1_safe = safe_name(
        "D1 AI & machine learning"
    )

    got = bool(
        tech_evidence[
            f"TECH_{d1_safe}"
        ].iloc[0]
    )

    if got:

        failures.append(
            "\nTECHNOLOGY ABSTRACT EXCLUSION\n"
            "D1 should NOT classify from abstract alone.\n"
            "Expected : False\n"
            f"Got      : {got}"
        )

    # =========================================================================
    # FINAL TEST RESULT
    # =========================================================================

    if failures:

        raise AssertionError(
            "\nSELF-TEST FAILED:\n"
            + "\n".join(failures)
        )

    print(
        "SELF-TEST PASSED."
    )

    print(
        f"Technology tests : {len(technology_tests)}"
    )

    print(
        f"Policy tests     : {len(policy_tests)}"
    )

    print(
        f"Environment tests: {len(environment_tests)}"
    )

    print(
        f"P3 negatives     : {len(p3_negative_tests)}"
    )

    print(
        "Field/abstract tests: PASSED"
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    path = SOURCE_FILE if SOURCE_FILE and os.path.isfile(SOURCE_FILE) else os.path.join(BASE, INFILE)

    if not os.path.exists(path):

        alternatives = [

            "tech_country_perpaper.csv",

            "tech_country_perpaper_INDEXED.csv",

            "extract_tech_country_perpaper.csv",

            "tech_country_perpaper_final.csv",
        ]

        for filename in alternatives:

            candidate = os.path.join(
                BASE,
                filename
            )

            if os.path.exists(candidate):

                path = candidate
                break

    if not os.path.exists(path):

        available = []

        if os.path.exists(BASE):

            available = os.listdir(
                BASE
            )

        raise FileNotFoundError(
            "\nINPUT FILE NOT FOUND\n"
            "----------------------------------------\n"
            f"Folder:\n{BASE}\n\n"
            f"Expected file:\n{INFILE}\n\n"
            f"Files found:\n{available}"
        )

    print("\n")
    print("=" * 80)
    print("READING INPUT")
    print("=" * 80)

    print(
        f"Input file: {path}"
    )

    df = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig"
    )

    print(
        f"Records: {len(df):,}"
    )

    return df, path


# =============================================================================
# AUDIT
# =============================================================================

def make_audit(
    df,
    columns,
    tech_flags,
    policy_flags,
    env_flags,
    tech_evidence,
    policy_evidence,
    env_evidence
):

    audit = pd.DataFrame(
        index=df.index
    )

    # -------------------------------------------------------------------------
    # Basic identifiers
    # -------------------------------------------------------------------------

    audit["Title"] = df[
        columns["title"]
    ]

    if columns["year"] is not None:

        audit["Year"] = df[
            columns["year"]
        ]

    # -------------------------------------------------------------------------
    # Layer counts
    # -------------------------------------------------------------------------

    audit["technology_n"] = (
        tech_flags.sum(axis=1)
    )

    audit["policy_n"] = (
        policy_flags.sum(axis=1)
    )

    audit["environment_n"] = (
        env_flags.sum(axis=1)
    )

    # -------------------------------------------------------------------------
    # Layer classification
    # -------------------------------------------------------------------------

    audit["technology_classified"] = (
        audit["technology_n"] > 0
    )

    audit["policy_classified"] = (
        audit["policy_n"] > 0
    )

    audit["environment_classified"] = (
        audit["environment_n"] > 0
    )

    audit["classified_any_layer"] = (
        audit["technology_classified"]
        |
        audit["policy_classified"]
        |
        audit["environment_classified"]
    )

    audit["classified_all_three_layers"] = (
        audit["technology_classified"]
        &
        audit["policy_classified"]
        &
        audit["environment_classified"]
    )

    audit["fully_unclassified"] = (
        ~audit["classified_any_layer"]
    )

    # -------------------------------------------------------------------------
    # Layer labels
    # -------------------------------------------------------------------------

    audit["technology_domains"] = (
        build_multilabel(
            tech_flags
        )
    )

    audit["policy_priorities"] = (
        build_multilabel(
            policy_flags
        )
    )

    audit["environmental_outcomes"] = (
        build_multilabel(
            env_flags
        )
    )

    # -------------------------------------------------------------------------
    # Field availability
    # -------------------------------------------------------------------------

    title = clean_text_series(
        df[
            columns["title"]
        ]
    )

    audit["has_title"] = (
        title.str.strip().ne("")
    )

    if columns["author_keywords"] is not None:

        author = clean_text_series(
            df[
                columns["author_keywords"]
            ]
        )

    else:

        author = empty_series(
            df.index
        )

    audit["has_author_keywords"] = (
        author.str.strip().ne("")
    )

    if columns["indexed_keywords"] is not None:

        indexed = clean_text_series(
            df[
                columns["indexed_keywords"]
            ]
        )

    else:

        indexed = empty_series(
            df.index
        )

    audit["has_indexed_keywords"] = (
        indexed.str.strip().ne("")
    )

    if columns["abstract"] is not None:

        abstract = clean_text_series(
            df[
                columns["abstract"]
            ]
        )

    else:

        abstract = empty_series(
            df.index
        )

    audit["has_abstract"] = (
        abstract.str.strip().ne("")
    )

    # -------------------------------------------------------------------------
    # Evidence totals
    # -------------------------------------------------------------------------

    audit["technology_title_evidence"] = (
        tech_evidence.filter(
            regex="_TITLE$"
        ).any(axis=1)
    )

    audit["technology_author_evidence"] = (
        tech_evidence.filter(
            regex="_AUTHOR$"
        ).any(axis=1)
    )

    audit["technology_indexed_evidence"] = (
        tech_evidence.filter(
            regex="_INDEXED$"
        ).any(axis=1)
    )

    audit["policy_title_evidence"] = (
        policy_evidence.filter(
            regex="_TITLE$"
        ).any(axis=1)
    )

    audit["policy_author_evidence"] = (
        policy_evidence.filter(
            regex="_AUTHOR$"
        ).any(axis=1)
    )

    audit["policy_indexed_evidence"] = (
        policy_evidence.filter(
            regex="_INDEXED$"
        ).any(axis=1)
    )

    audit["policy_abstract_evidence"] = (
        policy_evidence.filter(
            regex="_ABSTRACT$"
        ).any(axis=1)
    )

    audit["environment_title_evidence"] = (
        env_evidence.filter(
            regex="_TITLE$"
        ).any(axis=1)
    )

    audit["environment_author_evidence"] = (
        env_evidence.filter(
            regex="_AUTHOR$"
        ).any(axis=1)
    )

    audit["environment_indexed_evidence"] = (
        env_evidence.filter(
            regex="_INDEXED$"
        ).any(axis=1)
    )

    audit["environment_abstract_evidence"] = (
        env_evidence.filter(
            regex="_ABSTRACT$"
        ).any(axis=1)
    )

    return audit


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    tech_flags,
    policy_flags,
    env_flags
):

    rows = []

    # -------------------------------------------------------------------------
    # Technology
    # -------------------------------------------------------------------------

    for category in tech_flags.columns:

        rows.append({

            "Layer":
                "A — Technology",

            "Category":
                category,

            "Publications":
                int(
                    tech_flags[
                        category
                    ].sum()
                ),

            "Percent_of_corpus":
                100 *
                tech_flags[
                    category
                ].mean(),
        })

    # -------------------------------------------------------------------------
    # Policy
    # -------------------------------------------------------------------------

    for category in policy_flags.columns:

        rows.append({

            "Layer":
                "B — Policy",

            "Category":
                category,

            "Publications":
                int(
                    policy_flags[
                        category
                    ].sum()
                ),

            "Percent_of_corpus":
                100 *
                policy_flags[
                    category
                ].mean(),
        })

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    for category in env_flags.columns:

        rows.append({

            "Layer":
                "C — Environmental",

            "Category":
                category,

            "Publications":
                int(
                    env_flags[
                        category
                    ].sum()
                ),

            "Percent_of_corpus":
                100 *
                env_flags[
                    category
                ].mean(),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_summary(
    columns,
    tech_flags,
    policy_flags,
    env_flags
):

    print("\n")
    print("=" * 80)
    print("FINAL CLASSIFICATION SUMMARY")
    print("=" * 80)

    print("\nCLASSIFICATION DESIGN")
    print("-" * 80)

    print(
        "Technology : Title + Author Keywords + Indexed Keywords"
    )

    print(
        "Policy     : Title + Author Keywords + Indexed Keywords + Abstract"
    )

    print(
        "Environment: Title + Author Keywords + Indexed Keywords + Abstract"
    )

    print(
        "Threshold  : NONE"
    )

    print(
        "Weighting  : NONE"
    )

    print("\nINPUT FIELDS")
    print("-" * 80)

    for key, value in columns.items():

        print(
            f"{key:<22}: {value}"
        )

    # -------------------------------------------------------------------------
    # Technology
    # -------------------------------------------------------------------------

    print("\nLAYER A — TECHNOLOGY")
    print("-" * 80)

    for category in tech_flags.columns:

        count = int(
            tech_flags[
                category
            ].sum()
        )

        pct = (
            100 *
            tech_flags[
                category
            ].mean()
        )

        print(
            f"{category:<45}"
            f"{count:>7,}"
            f" ({pct:5.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Policy
    # -------------------------------------------------------------------------

    print("\nLAYER B — POLICY")
    print("-" * 80)

    for category in policy_flags.columns:

        count = int(
            policy_flags[
                category
            ].sum()
        )

        pct = (
            100 *
            policy_flags[
                category
            ].mean()
        )

        print(
            f"{category:<45}"
            f"{count:>7,}"
            f" ({pct:5.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    print("\nLAYER C — ENVIRONMENT")
    print("-" * 80)

    for category in env_flags.columns:

        count = int(
            env_flags[
                category
            ].sum()
        )

        pct = (
            100 *
            env_flags[
                category
            ].mean()
        )

        print(
            f"{category:<45}"
            f"{count:>7,}"
            f" ({pct:5.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------------------

    technology = tech_flags.any(
        axis=1
    )

    policy = policy_flags.any(
        axis=1
    )

    environment = env_flags.any(
        axis=1
    )

    any_layer = (
        technology
        |
        policy
        |
        environment
    )

    all_layers = (
        technology
        &
        policy
        &
        environment
    )

    none = ~any_layer

    print("\nCROSS-LAYER COVERAGE")
    print("-" * 80)

    print(
        f"At least one layer classified : "
        f"{int(any_layer.sum()):,} "
        f"({100 * any_layer.mean():5.2f}%)"
    )

    print(
        f"No layer classified           : "
        f"{int(none.sum()):,} "
        f"({100 * none.mean():5.2f}%)"
    )

    print(
        f"All three layers classified   : "
        f"{int(all_layers.sum()):,} "
        f"({100 * all_layers.mean():5.2f}%)"
    )


CANONICAL = '''Afghanistan;Albania;Algeria;Andorra;Argentina;Armenia;Australia;Austria;Azerbaijan;Bahrain;Bangladesh;Barbados;Belarus;Belgium;Benin;Bhutan;Bolivia;Bosnia and Herzegovina;Botswana;Brazil;Brunei;Bulgaria;Burkina Faso;Cambodia;Cameroon;Canada;Cape Verde;Central African Republic;Chad;Chile;China;Colombia;Comoros;Congo;Costa Rica;Cote d'Ivoire;Croatia;Cuba;Cyprus;Czechia;Denmark;Ecuador;Egypt;El Salvador;Estonia;Ethiopia;Fiji;Finland;France;Gabon;Georgia;Germany;Ghana;Greece;Guatemala;Guinea;Honduras;Hong Kong;Hungary;Iceland;India;Indonesia;Iran;Iraq;Ireland;Israel;Italy;Japan;Jordan;Kazakhstan;Kenya;Kuwait;Kyrgyzstan;Laos;Latvia;Lebanon;Lesotho;Libya;Lithuania;Luxembourg;Macau;Madagascar;Malawi;Malaysia;Maldives;Mali;Malta;Mauritania;Mauritius;Mexico;Moldova;Monaco;Mongolia;Montenegro;Morocco;Mozambique;Myanmar;Namibia;Nepal;Netherlands;New Zealand;Nigeria;North Macedonia;Norway;Oman;Pakistan;Palestine;Panama;Paraguay;Peru;Philippines;Poland;Portugal;Puerto Rico;Qatar;Romania;Russia;Rwanda;Saudi Arabia;Senegal;Serbia;Sierra Leone;Singapore;Slovakia;Slovenia;Somalia;South Africa;South Korea;South Sudan;Spain;Sri Lanka;Sudan;Sweden;Switzerland;Syria;Taiwan;Tajikistan;Tanzania;Thailand;Trinidad and Tobago;Tunisia;Turkey;Uganda;Ukraine;United Arab Emirates;United Kingdom;United States;Uruguay;Uzbekistan;Venezuela;Vietnam;Yemen;Zambia;Zimbabwe'''.split(';')
ALIASES = {
 'USA':'United States','U.S.A.':'United States','U.S.':'United States','US':'United States','United States of America':'United States',
 'UK':'United Kingdom','U.K.':'United Kingdom','Great Britain':'United Kingdom','England':'United Kingdom','Scotland':'United Kingdom','Wales':'United Kingdom','Northern Ireland':'United Kingdom',
 'Peoples R China':'China','People’s Republic of China':'China','People Republic of China':'China','P.R. China':'China','PR China':'China',
 'Republic of Korea':'South Korea','Korea':'South Korea','Korea, Republic of':'South Korea',
 'Russian Federation':'Russia','Viet Nam':'Vietnam','Lao PDR':'Laos','Czech Republic':'Czechia','Türkiye':'Turkey',
 'Hong Kong SAR':'Hong Kong','Hong Kong, China':'Hong Kong','Macao':'Macau','Ivory Coast':"Cote d'Ivoire",'Côte d’Ivoire':"Cote d'Ivoire",
 'The Netherlands':'Netherlands','UAE':'United Arab Emirates','U Arab Emirates':'United Arab Emirates','United Arab Emirate':'United Arab Emirates','Iran, Islamic Republic of':'Iran','Taiwan, China':'Taiwan',
 'DR Congo':'Congo','Democratic Republic of the Congo':'Congo','Republic of the Congo':'Congo',
 'Cabo Verde':'Cape Verde','Macedonia':'North Macedonia','Republic of Moldova':'Moldova'
}
FORMS={x:x for x in CANONICAL}; FORMS.update(ALIASES)
def regex_form(s): return re.escape(s).replace(r'\ ',r'\s+')
# One compiled expression is much faster than trying every country against every field.
COUNTRY_RE=re.compile(r'(?<![A-Za-z])('+'|'.join(regex_form(x) for x in sorted(FORMS,key=len,reverse=True))+r')(?![A-Za-z])',re.I)
LOOKUP={x.casefold():v for x,v in FORMS.items()}
def extract(text):
 out=[]
 if not isinstance(text, str):
  text = '' if pd.isna(text) else str(text)
 for m in COUNTRY_RE.finditer(text):
  c=LOOKUP[m.group(1).casefold()]
  if c not in out: out.append(c)
 return out

def repair_country_columns(df):
    if "Affiliations" not in df.columns or "Correspondence Address" not in df.columns:
        return df
    countries_out=[]; sources_out=[]
    for _, row in df.iterrows():
        aff=extract(row.get("Affiliations", "")); corr=extract(row.get("Correspondence Address", ""))
        old_value = row.get("countries", "")
        old_text = "" if pd.isna(old_value) else str(old_value)
        old=[x.strip() for x in old_text.split(";") if x.strip() and x.strip().casefold() not in {"nan", "none", "null"}]
        if aff: countries, source = aff, "affiliation"
        elif corr: countries, source = corr, "correspondence address"
        elif old: countries, source = old, "existing value (unmatched source text)"
        else: countries, source = [], "unresolved"
        countries_out.append("; ".join(countries)); sources_out.append(source)
    df["countries"] = countries_out
    df["n_countries"] = [len([x for x in c.split("; ") if x]) for c in countries_out]
    df["country_source"] = sources_out
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    global BASE, INFILE, SOURCE_FILE
    parser = argparse.ArgumentParser(description="Classify agriculture-technology papers into technology, policy, production, and environmental categories.")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to the input file's directory)")
    args = parser.parse_args()
    input_path = os.path.abspath(args.input_csv)
    SOURCE_FILE = input_path
    INFILE = os.path.basename(input_path)
    BASE = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(input_path)

    print("\n")
    print("=" * 80)
    print("AGRICULTURE 4.0 — THREE-LAYER CLASSIFICATION")
    print("FINAL V4 — DIRECT MATCH + ABSTRACT FOR POLICY/ENVIRONMENT")
    print("=" * 80)

    # =========================================================================
    # SELF TEST
    # =========================================================================

    self_test()

    # =========================================================================
    # LOAD
    # =========================================================================

    df, input_path = load_data()

    # This script may be rerun on a previously classified export. Remove only
    # generated classification/evidence columns before rebuilding them.
    generated_prefixes = ("TECH_", "POLICY_", "ENV_")
    generated_exact = {
        "tech_domains", "tech_n", "tech_primary", "policy_priorities", "policy_n",
        "environmental_outcomes", "environment_n", "classified_any_layer",
        "classified_all_three_layers", "tech_classification_basis",
        "production_category", "production_n"
    }
    df = df.drop(columns=[c for c in df.columns if c.startswith(generated_prefixes) or c in generated_exact], errors="ignore")

    # =========================================================================
    # DETECT COLUMNS
    # =========================================================================

    columns = detect_columns(
        df
    )

    print("\nDetected columns")
    print("-" * 80)

    for key, value in columns.items():

        print(
            f"{key:<22}: {value}"
        )

    if columns["author_keywords"] is None:

        print(
            "\nWARNING:"
            "\nAuthor Keywords column was not found."
            "\nAuthor keyword evidence will be unavailable."
        )

    if columns["indexed_keywords"] is None:

        print(
            "\nWARNING:"
            "\nIndexed Keywords column was not found."
            "\nIndexed keyword evidence will be unavailable."
        )

    if columns["abstract"] is None:

        print(
            "\nWARNING:"
            "\nAbstract column was not found."
            "\nPolicy/environment abstract evidence will be unavailable."
        )

    # =========================================================================
    # PREPARE TEXT
    # =========================================================================

    text = prepare_text(
        df,
        columns
    )

    # =========================================================================
    # LAYER A — TECHNOLOGY
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("LAYER A — TECHNOLOGY DOMAINS")
    print("=" * 80)

    tech_evidence = classify_dictionary(
        df,
        TECH_RX,
        text,
        include_abstract=False,
        prefix="TECH"
    )

    tech_flags = extract_final_flags(
        tech_evidence,
        TECH_RX.keys(),
        "TECH"
    )

    # Add all evidence columns to dataframe
    direct_tech_flags = tech_flags.copy()
    abstract_fallback = ~direct_tech_flags.any(axis=1)
    for category, rx in TECH_RX.items():
        safe = safe_name(category)
        abstract_hit = contains(rx, text["abstract"]) & abstract_fallback
        tech_evidence[f"TECH_{safe}_ABSTRACT"] = abstract_hit.astype(bool)
        tech_evidence[f"TECH_{safe}"] = (direct_tech_flags[category] | abstract_hit).astype(bool)
    tech_flags = extract_final_flags(tech_evidence, TECH_RX.keys(), "TECH")

    df = pd.concat(
        [df, tech_evidence],
        axis=1
    )

    df["tech_classification_basis"] = "title/keywords"
    df.loc[abstract_fallback & tech_flags.any(axis=1), "tech_classification_basis"] = "abstract fallback"
    df.loc[~tech_flags.any(axis=1), "tech_classification_basis"] = "unclassified"

    df["tech_domains"] = build_multilabel(
        tech_flags
    )

    df["tech_n"] = tech_flags.sum(
        axis=1
    )

    df["tech_primary"] = (
        determine_primary_technology(
            tech_flags
        )
    )

    # =========================================================================
    # LAYER B — POLICY
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("LAYER B — POLICY PRIORITIES")
    print("=" * 80)

    policy_evidence = classify_dictionary(
        df,
        POLICY_RX,
        text,
        include_abstract=True,
        prefix="POLICY"
    )

    policy_flags = extract_final_flags(
        policy_evidence,
        POLICY_RX.keys(),
        "POLICY"
    )

    df = pd.concat(
        [
            df,
            policy_evidence
        ],
        axis=1
    )

    df["policy_priorities"] = (
        build_multilabel(
            policy_flags
        )
    )

    df["policy_n"] = (
        policy_flags.sum(
            axis=1
        )
    )

    # Production is exposed separately as the existing P1 productivity/food-security priority.
    p1_code = "P1 Productivity & food security"
    df["production_category"] = policy_flags[p1_code].map({True: p1_code, False: ""})
    df["production_n"] = policy_flags[p1_code].astype(int)

    # =========================================================================
    # LAYER C — ENVIRONMENT
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("LAYER C — ENVIRONMENTAL OUTCOMES")
    print("=" * 80)

    env_evidence = classify_dictionary(
        df,
        ENV_RX,
        text,
        include_abstract=True,
        prefix="ENV"
    )

    env_flags = extract_final_flags(
        env_evidence,
        ENV_RX.keys(),
        "ENV"
    )

    df = pd.concat(
        [
            df,
            env_evidence
        ],
        axis=1
    )

    df["environmental_outcomes"] = (
        build_multilabel(
            env_flags
        )
    )

    df["environment_n"] = (
        env_flags.sum(
            axis=1
        )
    )

    # =========================================================================
    # CROSS-LAYER CLASSIFICATION
    # =========================================================================

    df["classified_any_layer"] = (
        tech_flags.any(axis=1)
        |
        policy_flags.any(axis=1)
        |
        env_flags.any(axis=1)
    )

    df["classified_all_three_layers"] = (
        tech_flags.any(axis=1)
        &
        policy_flags.any(axis=1)
        &
        env_flags.any(axis=1)
    )

    # =========================================================================
    # AUDIT
    # =========================================================================

    audit = make_audit(
        df,
        columns,
        tech_flags,
        policy_flags,
        env_flags,
        tech_evidence,
        policy_evidence,
        env_evidence
    )

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================

    print_summary(
        columns,
        tech_flags,
        policy_flags,
        env_flags
    )

    # =========================================================================
    # WRITE OUTPUTS
    # =========================================================================

    if WRITE:

        df = repair_country_columns(df)
        print("\n")
        print("=" * 80)
        print("WRITING OUTPUTS")
        print("=" * 80)

        os.makedirs(
            BASE,
            exist_ok=True
        )

        classified_path = os.path.join(
            BASE,
            OUT_CLASSIFIED
        )

        p3_path = os.path.join(
            BASE,
            OUT_P3
        )

        audit_path = os.path.join(
            BASE,
            OUT_AUDIT
        )

        unclassified_path = os.path.join(
            BASE,
            OUT_UNCLASSIFIED
        )

        summary_path = os.path.join(
            BASE,
            OUT_SUMMARY
        )

        # ---------------------------------------------------------------------
        # 1. FULL CLASSIFICATION
        # ---------------------------------------------------------------------

        df.to_csv(
            classified_path,
            index=False,
            encoding="utf-8-sig"
        )

        # ---------------------------------------------------------------------
        # 2. P3 CORPUS
        # ---------------------------------------------------------------------

        p3_column = (
            "POLICY_P3_Climate_resilience_and_risk_management"
        )

        if p3_column not in df.columns:

            raise KeyError(
                "\nP3 output column was not found:\n"
                f"{p3_column}"
            )

        df[
            df[p3_column]
        ].to_csv(
            p3_path,
            index=False,
            encoding="utf-8-sig"
        )

        # ---------------------------------------------------------------------
        # 3. AUDIT
        # ---------------------------------------------------------------------

        audit.to_csv(
            audit_path,
            index=False,
            encoding="utf-8-sig"
        )

        # ---------------------------------------------------------------------
        # 4. UNCLASSIFIED PAPERS
        # ---------------------------------------------------------------------

        unclassified = df[
            ~df["classified_any_layer"]
        ].copy()

        unclassified.to_csv(
            unclassified_path,
            index=False,
            encoding="utf-8-sig"
        )

        # ---------------------------------------------------------------------
        # 5. SUMMARY
        # ---------------------------------------------------------------------

        summary = build_summary(
            tech_flags,
            policy_flags,
            env_flags
        )

        summary.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig"
        )

        # ---------------------------------------------------------------------
        # REPORT
        # ---------------------------------------------------------------------

        print("\nFiles written:")
        print(
            f"  {classified_path}"
        )

        print(
            f"  {p3_path}"
        )

        print(
            f"  {audit_path}"
        )

        print(
            f"  {unclassified_path}"
        )

        print(
            f"  {summary_path}"
        )

        print("\nOutput counts:")

        print(
            f"  Full corpus          : "
            f"{len(df):,}"
        )

        print(
            f"  P3 corpus            : "
            f"{int(df[p3_column].sum()):,}"
        )

        print(
            f"  Fully unclassified   : "
            f"{len(unclassified):,}"
        )

        print(
            f"  Any-layer classified : "
            f"{int(df['classified_any_layer'].sum()):,}"
        )

        print(
            f"  All-three classified : "
            f"{int(df['classified_all_three_layers'].sum()):,}"
        )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()
