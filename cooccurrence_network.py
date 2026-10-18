# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 00:50:58 2026

"""

"""
Co-occurrence Network — Flexible Source Selection + CSV Export + Paper Mapping
==============================================================================
Sources you can toggle (any combination):
  • Title     bi/trigrams
  • Abstract  bi/trigrams
  • Author Keywords
  • Index Keywords
 
Outputs per time-window
  • cooccurrence_{window}.png / .pdf          — network plot
  • cooccurrence_combined.png / .pdf          — 4-panel plot
  • keyword_freq_{window}.csv                 — every keyword, freq, paper IDs & titles
  • cooccurrence_top_{window}.csv             — top pairs, co-occ count, paper IDs & titles
  • paper_keyword_map_{window}.csv            — every paper → its keyword list
"""
 
import pandas as pd
import numpy as np
import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations
import networkx as nx
from adjustText import adjust_text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run keyword co-occurrence network analysis.")
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="Path to the cleaned/classified bibliographic CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cooccurrence"),
                        help="Directory for generated figures and CSV outputs (default: outputs/cooccurrence).")
    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f"Input CSV does not exist: {args.input_csv}")
    OUT = str(args.output_dir)
    os.makedirs(OUT, exist_ok=True)

 
    # ═══════════════════════════════════════════════════════════════════════════════
    # ★  OUTPUT FOLDER  ★
    # ═══════════════════════════════════════════════════════════════════════════════
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # ★  USER CONFIGURATION — CHANGE THESE AS NEEDED  ★
    # ═══════════════════════════════════════════════════════════════════════════════
 
    # ---------- Source selection ----------
    # Set any combination of True / False
    SOURCE_CONFIG = {
        'use_title':           True,   # bi/trigrams from the Title column
        'use_abstract':        False,   # bi/trigrams from the Abstract column
        'use_author_keywords': True,   # semicolon-delimited Author Keywords column
        'use_index_keywords':  True,   # semicolon-delimited Index Keywords column
    }
 
    # ---------- N-gram settings (for Title / Abstract text sources) ----------
    NGRAM_RANGE = (2, 3)   # (2,2) = bigrams only | (2,3) = bi + trigrams | (3,3) = trigrams only
    NGRAM_MIN_DF = 50      # phrase must appear in at least this many documents globally
 
    # ---------- Time windows ----------
    WINDOWS = [
        ('2010–2013', 2010, 2013),
        ('2014–2017', 2014, 2017),
        ('2018–2021', 2018, 2021),
        ('2022–2025', 2022, 2025),
    ]
 
    # ---------- Network thresholds per window size ----------
    # Format: {max_docs: (min_keyword_freq, min_edge_cooccurrence)}
    NETWORK_THRESHOLDS = {
        800:   (8,  3),
        3000:  (15, 5),
        8000:  (25, 8),
        99999: (40, 12),
    }
 
    # ---------- Community detection ----------
    COMMUNITY_METHOD = 'leiden'   # 'louvain' or 'leiden'
    RESOLUTION       = None       # float or None for adaptive
    ADAPTIVE_RESOLUTION = {
        800:   1.0,
        3000:  1.2,
        8000:  1.5,
        99999: 2.0,
    }
 
    # ---------- CSV export ----------
    TOP_N_PAIRS    = 200   # how many top co-occurrence pairs to write to CSV
    TOP_N_KEYWORDS = 300   # how many top keywords to include in keyword_freq CSV
    MAX_PAPERS_PER_ROW = 50  # cap paper list per CSV row to keep files manageable
                              # set to None for no cap
 
    # ---------- Paper identifier column ----------
    # The script will try these column names in order; falls back to row index
    PAPER_ID_COLS  = ['DOI', 'UT (Unique WOS ID)', 'EID', 'AN']
    PAPER_TTL_COL  = 'Title'   # column used for human-readable paper title in CSV
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # NORMALIZATION TABLES  (same as original script — edit freely)
    # ═══════════════════════════════════════════════════════════════════════════════
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError as exc:
        raise SystemExit("NLTK English stopwords are missing. Run: python -m nltk.downloader stopwords") from exc
 
    OBVIOUS = {
        'study','studies','research','analysis','approach','method','methods',
        'system','systems','model','models','review','paper','results','result',
        'based','using','use','used','new','novel','proposed','data',
        'performance','development','application','applications','technique',
        'techniques','technology','technologies','framework','design','process',
        'management','impact','effect','effects','evaluation','experimental',
        'optimization','monitoring','detection','estimation','prediction',
        'assessment','implementation','integration','strategy','challenges',
        'future','literature','survey','overview','field','area','areas',
        'case','problem','solution','solutions','state','current','different',
        'various','high','low','large','small','improved','efficient',
        'effective','important','significant','increase','decrease','change',
        'changes','level','levels','factors','factor','role','potential',
        'information','quality','conditions','condition','feature','features',
        'value','values','time','type','types','test','testing','article',
        'agriculture','agricultural','farming','farm','farms','crop','crops',
        'plant','plants','food','production','yield','growth','agri','agric',
        'agronomic','agronomy','cultivation','cultivar','cultivars',
        'sustainability','sustainable','environment','environmental',
        'smart agricultures','modern agricultures',
        'proposed method','proposed model','proposed system','proposed approach',
        'experimental results','results show','results showed','paper presents',
        'study aims','study investigates',
        'rights reserved','elsevier right','elsevier rights reserved',
        'published elsevier','elsevier ltd','ltd right','ltd rights reserved',
        'open access','creativecommons org','creativecommons org licens',
        'access cc','license http creativecommon','http creativecommons org',
        'org licens','licens http','http creativecommon',
        'recent year','recent years','wide range','become increasingly',
        'fields including','work present','also provide','presents comprehensive',
        'provides valuable','provide valuable','valuable insight','offering valuable',
        'findings provide','findings highlight','findings suggest',
        'plays vital','ensuring security','compared conventional',
        'compared traditional','labor intensive','manual labor','labour intensive',
        'consuming labor','consuming labor intensive','intensive consuming',
        'labor intensive consuming','long term','short term',
        'cutting edge','among others','advancing precision',
        'precision precision','precision enabling','precision practice',
        'smart precision','learning precision','machine learning precision',
        'open access cc','license http','cc nc','nc nd license','access cc nc',
        'nd license','license http creativecommon','licens http',
        'plays crucial','plays vital role','recent advancement','recent advances',
        'address issue','address challenge','across diverse','significantly reduce',
        'improve accuracy','accuracy compared','main objective','become increasingly',
        'increasing productivity','enhance precision','enhancing precision',
        'improve efficiency','improve productivity','ai powered',
        'deep learning precision','machine learning deep','learning classification',
        'machine learning classification','machine learning smart',
        'developing country','developing countries',
        'presents comprehensive review','comprehensive review',
        'pests diseas','pest disease',
        'aerial vehicles uav','aerial vehicle uav',
        'network wsn','iot smart','iot device',
        'cc nc nd','licenses nc nd','org licenses nc','nd license http',
        'elsevier behalf','cc nc','nc nd','nd license',
        'however existing','challenging due','training dataset',
        'enhance accuracy','accuracy achieved','accuracy score',
        'accuracy reliability','enhance efficiency','improving efficiency',
        'technological advancement','learning iot','learning predict',
        'learning ml','precision pa','driven decision','data driven decision',
        'crop classification','real health','health soil','plant monitoring',
        'plant health','crop health','informed decision',
        'resource allocation','identification',
    }

    # Keywords that must NEVER be auto-merged into a longer form.
    # Add any term you want to keep as its own node in the network.
    PROTECT = {
        'neural network',
        'machine learning',
        'deep learning',    
        'remote sensing',
        'random forest',
        'support vector machine',
        'internet of things',
        'precision agriculture',
        'image processing',
        'sensor network',
        'decision tree',
        'climate change',
        'soil moisture',
    }
 
    NORM = {
        'agricultural':'agriculture','agric':'agriculture','agri':'agriculture',
        'smart agricultures':'smart agriculture','modern agricultures':'smart agriculture',
        'machine-learning':'machine learning','deep-learning':'deep learning',
        'reinforcement-learning':'reinforcement learning','transfer-learning':'transfer learning',
        'e-agriculture':'digital agriculture',
        'iot':'internet of things','ml':'machine learning',
        'ai':'artificial intelligence','dl':'deep learning',
        'cnn':'convolutional neural network','cnns':'convolutional neural network',
        'convolutional neural networks':'convolutional neural network',
        'ann':'artificial neural network','anns':'artificial neural network',
        'artificial neural networks':'artificial neural network',
        'uav':'unmanned aerial vehicle','uavs':'unmanned aerial vehicle',
        'unmanned aerial vehicles':'unmanned aerial vehicle',
        'unmanned aerial vehicles Uav':'unmanned aerial vehicle',
        'drone':'unmanned aerial vehicle','drones':'unmanned aerial vehicle',
        'gis':'geographic information system',
        'ndvi':'normalized difference vegetation index',
        'vegetation indexes':'vegetation index','vegetation indices':'vegetation index',
        'rf':'random forest','random forests':'random forest',
        'svm':'support vector machine','svms':'support vector machine',
        'support vector machines':'support vector machine',
        'wsn':'wireless sensor network','wsns':'wireless sensor network',
        'wireless sensor networks':'wireless sensor network',
        'gnss':'global navigation satellite system',
        'gps':'global positioning system',
        'lidar':'light detection and ranging',
        'pa':'precision agriculture','precision farming':'precision agriculture',
        'smart farming':'smart agriculture','rs':'remote sensing',
        'ghg':'greenhouse gas','greenhouse gases':'greenhouse gas',
        'co2':'carbon dioxide','n2o':'nitrous oxide',
        'nue':'nitrogen use efficiency','wue':'water use efficiency',
        'et':'evapotranspiration',
        'neural networks':'neural network','neural-networks':'neural network',
        'neural-network':'neural network',
        'sensors':'sensor','robots':'robot','algorithms':'algorithm',
        'images':'image','datasets':'dataset','simulations':'simulation',
        'classifications':'classification','predictions':'prediction',
        'recommendations':'recommendation',
        'plants botany':'botany','classification of information':'classification',
        'learning algorithms':'learning algorithm','learning systems':'learning system',
        'particle swarm optimization pso':'particle swarm optimization',
        'swarm optimization':'particle swarm optimization',
        'particle swarm':'particle swarm optimization',
        'genetic algorithms':'genetic algorithm','decision trees':'decision tree',
        'recurrent neural networks':'recurrent neural network',
        'long short-term memory':'lstm',
        'soil moisture sensors':'soil moisture sensor',
        'disease detection':'plant disease detection',
        'pest detection':'pest management',
        'weed detection':'weed management','weed control':'weed management',
        'water management':'water resource management',
        'water resources':'water resource management',
        'climate-change':'climate change','global warming':'climate change',
        'greenhouse effect':'climate change',
        'soils':'soil','fertilizers':'fertilizer',
        'nutrients':'nutrient management',
        'internet of things iot':'internet of things',
        'internet of thing':'internet of things',
        'objects detection':'object detection','decisions makings':'decision making',
        'decision supports':'decision support system',
        'crop yield':'yield prediction','crop yield prediction':'yield prediction',
        'yield estimation':'yield prediction',
        'crop monitoring':'crop health monitoring',
        'image-analysis':'image analysis','image analyze':'image analysis',
        'climate resilient':'climate resilience','climate-resilient':'climate resilience',
        'climate resilient agriculture':'climate resilient agriculture',
        'climate-resilient agriculture':'climate resilient agriculture',
        'climate smart':'climate smart agriculture',
        'climate-smart':'climate smart agriculture',
        'climate-smart agriculture':'climate smart agriculture',
        'drought resistant':'drought resistance','drought-resistant':'drought resistance',
        'drought tolerant':'drought tolerance','drought-tolerant':'drought tolerance',
        'internet thing':'internet of things','internet things':'internet of things',
        'things iot':'internet of things','internet things iot':'internet of things',
        'internet of thing iot':'internet of things','iot internet thing':'internet of things',
        'iot internet':'internet of things',
        'sensor networks':'sensor network',
        'sensor networks wsn':'wireless sensor network',
        'sensor network wsn':'wireless sensor network',
        'networks wireless sensor':'wireless sensor network',
        'sensor networks wireless':'wireless sensor network',
        'wireless sensor networks wsn':'wireless sensor network',
        'decision support systems':'decision support system',
        'decision support':'decision support system',
        'unmanned aerial':'unmanned aerial vehicle',
        'aerial vehicle':'unmanned aerial vehicle','aerial vehicles':'unmanned aerial vehicle',
        'vehicles uav':'unmanned aerial vehicle','vehicle uav':'unmanned aerial vehicle',
        'uav unmanned aerial':'unmanned aerial vehicle',
        'unmanned aerial vehicle uav':'unmanned aerial vehicle',
        'unmanned aircraft':'unmanned aerial vehicle',
        'robotic':'robotics','agricultural robot':'agricultural robotics',
        'agricultural robots':'agricultural robotics',
        'intelligent robot':'agricultural robotics',
        'convolutional neural':'convolutional neural network',
        'network cnn':'convolutional neural network',
        'neural network cnn':'convolutional neural network',
        'deep learning dl':'deep learning','learning dl':'deep learning',
        'learning deep':'deep learning',
        'machine learning ml':'machine learning','learning machine':'machine learning',
        'learning model':'machine learning','learning algorithm':'machine learning algorithm',
        'learning smart':'smart agriculture','learning system':'machine learning',
        'learning driven':'machine learning',
        'intelligence ai':'artificial intelligence',
        'artificial intelligence ai':'artificial intelligence',
        'ai driven':'artificial intelligence',
        'support vector':'support vector machine','vector machine':'support vector machine',
        'difference vegetation':'vegetation index',
        'difference vegetation index':'vegetation index',
        'normalized difference vegetation':'vegetation index',
        'normalized difference vegetation index':'vegetation index',
        'moisture sensor':'soil moisture sensor','soil moisture sensor':'soil moisture',
        'iot sensor':'internet of things','term memory':'long short term memory',
        'images processing':'image processing','pre processing':'preprocessing',
        'error rmse':'root mean square error','mean square':'mean square error',
        'square error':'mean square error','root mean':'root mean square error',
        'precision recall':'classification metrics','accuracy rate':'accuracy','plant diseas':'plant disease',
        'diseas':'disease','unmanned aerial vehicles uav':'unmanned aerial vehicle',
        'higher accuracy':'accuracy','network architecture':'neural network architecture','Plant Diseas':'Plant Disease','Agricultural Robotic':'Agricultural Robotics' 
    }
 
 
    def _lemma_phrase(phrase):
        words = phrase.split()
        if not words: return phrase
        last = words[-1]
        if len(last) <= 3 or last in {'analysis','basis','thesis','diagnosis','stress',
                                       'loss','biomass','access','process','progress',
                                       'wireless','less','across','class','grass','gas',
                                       'focus'}:
            return phrase
        if last.endswith('ies') and len(last) > 4:   words[-1] = last[:-3]+'y'
        elif last.endswith('ses') or last.endswith('zes') or last.endswith('xes'):
            words[-1] = last[:-2]
        elif last.endswith('ves') and len(last) > 4:  words[-1] = last[:-3]+'f'
        elif last.endswith('s') and not last.endswith('ss') and not last.endswith('us'):
            words[-1] = last[:-1]
        return ' '.join(words)
 
 
    def clean_kw(kw):
        kw = kw.strip().lower()
        kw = re.sub(r'[^\w\s\-/]', '', kw)
        kw = re.sub(r'\s+', ' ', kw).strip()
        kw = kw.replace('-', ' ').strip()
        kw = re.sub(r'\s+', ' ', kw).strip()
        if kw in NORM: kw = NORM[kw]
        kw = _lemma_phrase(kw)
        if kw in NORM: kw = NORM[kw]
        return kw
 
 
    def is_valid(kw):
        return (kw and len(kw) > 2
                and kw not in stop_words
                and kw not in OBVIOUS
                and not all(c in '- /' for c in kw)
                and not kw.isdigit())
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ═══════════════════════════════════════════════════════════════════════════════
    print("Loading data …")
    df = pd.read_csv(args.input_csv, low_memory=False)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.reset_index(drop=True)  # ensure clean integer index
    print(f"  Records: {len(df)}, Years: {df['Year'].min():.0f}–{df['Year'].max():.0f}")
 
    # ── Resolve paper-ID column ──────────────────────────────────────────────────
    _pid_col = None
    for c in PAPER_ID_COLS:
        if c in df.columns:
            _pid_col = c
            break
    if _pid_col:
        print(f"  Paper identifier column: '{_pid_col}'")
    else:
        print("  Paper identifier column: row index (no DOI/UT column found)")
 
    def get_paper_id(row_idx):
        if _pid_col:
            v = df.at[row_idx, _pid_col]
            return str(v) if pd.notna(v) else str(row_idx)
        return str(row_idx)
 
    def get_paper_title(row_idx):
        if PAPER_TTL_COL in df.columns:
            v = df.at[row_idx, PAPER_TTL_COL]
            return str(v)[:120] if pd.notna(v) else ''
        return ''
 
    active_sources = [k for k, v in SOURCE_CONFIG.items() if v]
    print(f"  Active sources: {active_sources}")
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # BUILD N-GRAM MATRICES  (only for text sources that are enabled)
    # ═══════════════════════════════════════════════════════════════════════════════
    def _make_text_col(cols):
        text = pd.Series([''] * len(df))
        for c in cols:
            if c in df.columns:
                text = text + ' ' + df[c].fillna('').str.lower()
        text = text.str.strip()
        text = text.apply(lambda t: re.sub(r'[^a-z\s]', ' ', t))
        text = text.apply(lambda t: re.sub(r'\s+', ' ', t).strip())
        return text
 
    extra_stop = list(stop_words | OBVIOUS)
 
    # Pre-compute vectorisers once over the full corpus, then index per window
    X_title = X_abstract = None
    vocab_title = vocab_abstract = []
 
    if SOURCE_CONFIG['use_title'] and 'Title' in df.columns:
        print("Building title n-gram matrix …")
        _t = _make_text_col(['Title'])
        _vec = CountVectorizer(ngram_range=NGRAM_RANGE, min_df=NGRAM_MIN_DF,
                               max_df=0.5, stop_words=extra_stop, max_features=3000)
        X_title = _vec.fit_transform(_t)
        vocab_title = _vec.get_feature_names_out()
        print(f"  Title phrases: {len(vocab_title)}")
 
    if SOURCE_CONFIG['use_abstract'] and 'Abstract' in df.columns:
        print("Building abstract n-gram matrix …")
        _t = _make_text_col(['Abstract'])
        _vec = CountVectorizer(ngram_range=NGRAM_RANGE, min_df=NGRAM_MIN_DF,
                               max_df=0.5, stop_words=extra_stop, max_features=3000)
        X_abstract = _vec.fit_transform(_t)
        vocab_abstract = _vec.get_feature_names_out()
        print(f"  Abstract phrases: {len(vocab_abstract)}")
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # KEYWORD EXTRACTION  (returns keywords + paper-level tracking)
    # ═══════════════════════════════════════════════════════════════════════════════
    def extract_kws_for_row(idx, row):
        """Return set of valid cleaned keywords for a single paper."""
        kws = set()
        if SOURCE_CONFIG['use_author_keywords']:
            val = row.get('Author Keywords', '')
            if pd.notna(val) and str(val).strip():
                for p in str(val).split(';'):
                    k = clean_kw(p)
                    if is_valid(k): kws.add(k)
        if SOURCE_CONFIG['use_index_keywords']:
            val = row.get('Index Keywords', '')
            if pd.notna(val) and str(val).strip():
                for p in str(val).split(';'):
                    k = clean_kw(p)
                    if is_valid(k): kws.add(k)
        if X_title is not None:
            for c in X_title[idx].nonzero()[1]:
                k = clean_kw(vocab_title[c])
                if is_valid(k): kws.add(k)
        if X_abstract is not None:
            for c in X_abstract[idx].nonzero()[1]:
                k = clean_kw(vocab_abstract[c])
                if is_valid(k): kws.add(k)
        return kws
 
 
    print("Extracting keywords for all papers …")
    all_kw_lists = []
    for i in range(len(df)):
        all_kw_lists.append(extract_kws_for_row(i, df.iloc[i]))
    df['_kws'] = all_kw_lists
 
    flat_global = [k for ks in df['_kws'] for k in ks]
    kw_freq_global = Counter(flat_global)
    print(f"  Unique keywords (before dedup): {len(kw_freq_global)}")
    # Build global keyword → paper-index map for merge validation
    kw_to_papers_global = defaultdict(set)
    for i, ks in enumerate(df['_kws']):
        for k in ks:
            kw_to_papers_global[k].add(i)
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # AUTO DEDUPLICATION
    # ═══════════════════════════════════════════════════════════════════════════════
    # def build_auto_merge(freq, min_count=10):
    #     candidates = {k for k, c in freq.items() if c >= min_count and ' ' in k}
    #     merge = {}
    #     by_len = sorted(candidates, key=lambda x: (-len(x.split()), -freq[x]))
    #     word_sets = {k: set(k.split()) for k in candidates}
    #     for short in by_len:
    #         if short in merge: continue
    #         if short in PROTECT: continue
    #         s_words = word_sets[short]; s_freq = freq[short]
    #         best_target = None; best_freq = 0
    #         for long in by_len:
    #             if long == short or long in merge: continue
    #             l_words = word_sets[long]
    #             if len(l_words) <= len(s_words): continue
    #             diff = l_words - s_words
    #             if s_words.issubset(l_words) and len(diff) <= 2:
    #                 l_freq = freq[long]
    #                 if l_freq > best_freq:
    #                     best_target = long; best_freq = l_freq
    #         if best_target and best_freq >= s_freq * 0.3:
    #             if best_freq >= s_freq: merge[short] = best_target
    #             else: merge[best_target] = short
    #     all_kws = sorted(freq.keys(), key=lambda x: -freq[x])
    #     for kw in all_kws:
    #         if kw in merge: continue
    #         for variant in [kw+'s', kw+'es']:
    #             if variant in freq and variant not in merge and variant != kw:
    #                 if freq[variant] < freq[kw]: merge[variant] = kw
    #                 else: merge[kw] = variant; break
    #     return merge
 
    # AUTO_MERGE = build_auto_merge(kw_freq_global, min_count=10)
    # print(f"  Auto-merge rules: {len(AUTO_MERGE)}")
    def build_auto_merge(freq, kw_papers, min_count=10, min_jaccard=0.35):
        """
        Merge shorter keyword into longer only when:
          1. All words of the shorter appear in the longer (word-subset rule)
          2. The two keywords share enough papers (Jaccard ≥ min_jaccard)
             — prevents merging terms that describe genuinely different concepts
          3. The shorter term is not in PROTECT
        min_jaccard=0.35 means at least 35% of the combined paper pool
        must be shared before a merge is allowed.
        """
        candidates = {k for k, c in freq.items() if c >= min_count and ' ' in k}
        merge = {}
        by_len = sorted(candidates, key=lambda x: (-len(x.split()), -freq[x]))
        word_sets = {k: set(k.split()) for k in candidates}

        for short in by_len:
            if short in merge: continue
            if short in PROTECT: continue
            s_words = word_sets[short]
            s_freq  = freq[short]
            s_papers = kw_papers.get(short, set())
            if not s_papers: continue

            best_target = None
            best_freq   = 0

            for long in by_len:
                if long == short or long in merge: continue
                l_words = word_sets[long]
                if len(l_words) <= len(s_words): continue

                diff = l_words - s_words
                if not (s_words.issubset(l_words) and len(diff) <= 2):
                    continue

                # ── Jaccard similarity on paper sets ──────────────────────────
                l_papers = kw_papers.get(long, set())
                if not l_papers: continue
                intersection = len(s_papers & l_papers)
                union        = len(s_papers | l_papers)
                jaccard      = intersection / union if union else 0

                if jaccard < min_jaccard:
                    continue          # different concepts — do NOT merge

                l_freq = freq[long]
                if l_freq > best_freq:
                    best_target = long
                    best_freq   = l_freq

            if best_target and best_freq >= s_freq * 0.3:
                if best_freq >= s_freq:
                    merge[short] = best_target
                else:
                    merge[best_target] = short

        # Trailing-s dedup (sensors→sensor etc.) — also Jaccard-gated
        all_kws = sorted(freq.keys(), key=lambda x: -freq[x])
        for kw in all_kws:
            if kw in merge or kw in PROTECT: continue
            for variant in [kw+'s', kw+'es']:
                if variant in freq and variant not in merge and variant != kw:
                    vp = kw_papers.get(variant, set())
                    kp = kw_papers.get(kw, set())
                    if vp and kp:
                        j = len(vp & kp) / len(vp | kp)
                        if j < min_jaccard:
                            continue    # e.g. "gas" and "gases" — different papers
                    if freq[variant] < freq[kw]:
                        merge[variant] = kw
                    else:
                        merge[kw] = variant
                    break

        return merge



    AUTO_MERGE = build_auto_merge(kw_freq_global, kw_to_papers_global, min_count=10, min_jaccard=0.35)
    def apply_merge(kw_set, merge_dict):
        return {merge_dict.get(k, k) for k in kw_set}
 
    df['_kws'] = df['_kws'].apply(lambda ks: apply_merge(ks, AUTO_MERGE))
    flat_global = [k for ks in df['_kws'] for k in ks]
    kw_freq_global = Counter(flat_global)
    print(f"  Unique keywords (after dedup): {len(kw_freq_global)}")
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # MATLAB COLORS FOR COMMUNITIES
    # ═══════════════════════════════════════════════════════════════════════════════
    MATLAB_COLORS = [
        '#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30',
        '#4DBEEE','#A2142F','#FF6F61','#6B5B95','#88B04B',
        '#F7CAC9','#92A8D1','#955251','#B565A7','#009B77',
        '#DD4124','#45B8AC','#5B5EA6','#E15D44','#9B2335',
    ]
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # MAIN ANALYSIS PER WINDOW
    # ═══════════════════════════════════════════════════════════════════════════════
    def _fmt_paper_list(idx_list):
        """Convert list of row indices to readable ID + title strings (capped)."""
        ids, titles = [], []
        cap = MAX_PAPERS_PER_ROW if MAX_PAPERS_PER_ROW else len(idx_list)
        for i in idx_list[:cap]:
            ids.append(get_paper_id(i))
            titles.append(get_paper_title(i))
        suffix = f' … (+{len(idx_list)-cap} more)' if len(idx_list) > cap else ''
        return ' | '.join(ids) + suffix, ' | '.join(t for t in titles if t) + suffix
 
 
    def run_window(sub_df, label, min_freq, edge_min, resolution):
        print(f"\n{'='*60}\n  {label}  ({len(sub_df)} docs)\n{'='*60}")
        sub_idx   = sub_df.index.tolist()
        kw_lists  = sub_df['_kws'].tolist()
        flat      = [k for ks in kw_lists for k in ks]
        kw_freq   = Counter(flat)
 
        # ── Paper→keyword and keyword→papers maps ────────────────────────────────
        kw_to_papers  = defaultdict(set)   # keyword → set of row indices
        pair_to_papers = defaultdict(set)  # (kw_a, kw_b) sorted → set of row indices
 
        for row_i, ks in zip(sub_idx, kw_lists):
            for k in ks:
                kw_to_papers[k].add(row_i)
 
        freq_kws = {k for k, c in kw_freq.items() if c >= min_freq}
        print(f"  Frequent keywords (≥{min_freq}): {len(freq_kws)}")
        if len(freq_kws) < 15:
            print("  ⚠ Too few keywords, skipping"); return None
 
        cooccur = defaultdict(int)
        for row_i, ks in zip(sub_idx, kw_lists):
            filt = sorted({k for k in ks if k in freq_kws})
            for a, b in combinations(filt, 2):
                pair_to_papers[(a, b)].add(row_i)
                cooccur[(a, b)] += 1
 
        # ── Build graph ───────────────────────────────────────────────────────────
        G = nx.Graph()
        for (a, b), w in cooccur.items():
            if w >= edge_min:
                G.add_edge(a, b, weight=w)
        if len(G.nodes) < 10:
            print("  ⚠ Network too small"); return None
 
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        print(f"  Network: {len(G.nodes)} nodes, {len(G.edges)} edges")
 
        if len(G.edges) > 100_000:
            edge_threshold = sorted([d['weight'] for _,_,d in G.edges(data=True)],
                                     reverse=True)[min(80000, len(G.edges)-1)]
            G2 = nx.Graph()
            for u, v, d in G.edges(data=True):
                if d['weight'] >= edge_threshold:
                    G2.add_edge(u, v, weight=d['weight'])
            largest = max(nx.connected_components(G2), key=len)
            G = G2.subgraph(largest).copy()
            print(f"  Pruned to: {len(G.nodes)} nodes, {len(G.edges)} edges")
 
        # ── Community detection ───────────────────────────────────────────────────
        print(f"  Community method: {COMMUNITY_METHOD}, resolution={resolution}")
        if COMMUNITY_METHOD == 'leiden':
            import igraph as ig, leidenalg
            nodes_list = list(G.nodes())
            node_idx   = {n: i for i, n in enumerate(nodes_list)}
            edges      = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
            weights    = [G[u][v]['weight'] for u, v in G.edges()]
            G_ig = ig.Graph(n=len(nodes_list), edges=edges, directed=False)
            G_ig.vs['name'] = nodes_list; G_ig.es['weight'] = weights
            part = leidenalg.find_partition(
                G_ig, leidenalg.RBConfigurationVertexPartition,
                weights='weight', resolution_parameter=resolution, seed=42)
            comms = [set(G_ig.vs[m]['name'] for m in members) for members in part]
        else:
            from networkx.algorithms.community import louvain_communities
            comms = list(louvain_communities(G, weight='weight', resolution=resolution, seed=42))
 
        if len([c for c in comms if len(c) >= 3]) < 4:
            retry_res = resolution * 1.8
            print(f"  Retrying with resolution={retry_res:.1f}")
            if COMMUNITY_METHOD == 'leiden':
                part = leidenalg.find_partition(
                    G_ig, leidenalg.RBConfigurationVertexPartition,
                    weights='weight', resolution_parameter=retry_res, seed=42)
                comms = [set(G_ig.vs[m]['name'] for m in members) for members in part]
            else:
                comms = list(louvain_communities(G, weight='weight',
                                                 resolution=retry_res, seed=42))
 
        comms = [c for c in comms if len(c) >= 3]
        print(f"  Communities (≥3): {len(comms)}")
 
        node_comm = {}
        for i, c in enumerate(comms):
            for n in c: node_comm[n] = i
 
        # ── CSV 1 — keyword frequency + paper mapping ─────────────────────────────
        tag = label.replace('–', '-')
        kw_rows = []
        for kw, cnt in kw_freq.most_common(TOP_N_KEYWORDS):
            papers = sorted(kw_to_papers[kw])
            pid_str, ttl_str = _fmt_paper_list(papers)
            comm_id = node_comm.get(kw, -1)
            kw_rows.append({
                'Keyword':      kw,
                'Frequency':    cnt,
                'Paper_Count':  len(papers),
                'Community_ID': comm_id,
                'Paper_IDs':    pid_str,
                'Paper_Titles': ttl_str,
            })
        pd.DataFrame(kw_rows).to_csv(
            os.path.join(OUT, f'keyword_freq_{tag}.csv'), index=False)
        print(f"  ✓ keyword_freq_{tag}.csv  ({len(kw_rows)} keywords)")
 
        # ── CSV 2 — top co-occurrence pairs + paper mapping ───────────────────────
        pair_rows = []
        sorted_pairs = sorted(cooccur.items(), key=lambda x: -x[1])[:TOP_N_PAIRS]
        for (a, b), cnt in sorted_pairs:
            papers = sorted(pair_to_papers[(a, b)])
            pid_str, ttl_str = _fmt_paper_list(papers)
            comm_a = node_comm.get(a, -1)
            comm_b = node_comm.get(b, -1)
            same_comm = 'Yes' if comm_a == comm_b and comm_a != -1 else 'No'
            pair_rows.append({
                'Keyword_A':    a,
                'Keyword_B':    b,
                'Co_occurrence': cnt,
                'Paper_Count':  len(papers),
                'Same_Community': same_comm,
                'Comm_A': comm_a,
                'Comm_B': comm_b,
                'Paper_IDs':    pid_str,
                'Paper_Titles': ttl_str,
            })
        pd.DataFrame(pair_rows).to_csv(
            os.path.join(OUT, f'cooccurrence_top_{tag}.csv'), index=False)
        print(f"  ✓ cooccurrence_top_{tag}.csv  ({len(pair_rows)} pairs)")
 
        # ── CSV 3 — paper → keyword map ───────────────────────────────────────────
        paper_rows = []
        for row_i, ks in zip(sub_idx, kw_lists):
            top_kws = sorted(ks, key=lambda k: kw_freq.get(k, 0), reverse=True)
            paper_rows.append({
                'Row_Index':   row_i,
                'Paper_ID':    get_paper_id(row_i),
                'Paper_Title': get_paper_title(row_i),
                'Year':        df.at[row_i, 'Year'] if 'Year' in df.columns else '',
                'Keyword_Count': len(top_kws),
                'Keywords':    ' | '.join(top_kws),
            })
        pd.DataFrame(paper_rows).to_csv(
            os.path.join(OUT, f'paper_keyword_map_{tag}.csv'), index=False)
        print(f"  ✓ paper_keyword_map_{tag}.csv  ({len(paper_rows)} papers)")
 
        return {'G': G, 'comms': comms, 'node_comm': node_comm,
                'kw_freq': kw_freq, 'n_docs': len(sub_df), 'window': label}
 
 
    # ── Run all windows ───────────────────────────────────────────────────────────
    all_results = {}
    for wl, ys, ye in WINDOWS:
        sub = df[(df['Year'] >= ys) & (df['Year'] <= ye)]
        if len(sub) < 30: continue
 
        n = len(sub)
        for threshold, (mf, em) in sorted(NETWORK_THRESHOLDS.items()):
            if n < threshold: break
 
        if RESOLUTION is not None:
            rv = RESOLUTION
        else:
            for threshold, rv in sorted(ADAPTIVE_RESOLUTION.items()):
                if n < threshold: break
 
        r = run_window(sub, wl, mf, em, rv)
        if r: all_results[wl] = r
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # PLOTTING — individual co-occurrence network per window
    # ═══════════════════════════════════════════════════════════════════════════════
    def plot_cooccur(res, fp, top_n=50):
        G = res['G']; kf = res['kw_freq']
        comms = res['comms']; nc = res['node_comm']
        top = [k for k, _ in kf.most_common(300) if k in G.nodes][:top_n]
        sG  = G.subgraph(top).copy()
        sG.remove_nodes_from(list(nx.isolates(sG)))
        if len(sG.nodes) < 3: return
        print(f"  Co-occ viz [{res['window']}]: {len(sG.nodes)} nodes, {len(sG.edges)} edges")
 
        fig, ax = plt.subplots(figsize=(18, 15))
        ns    = [kf.get(n, 12) for n in sG.nodes]; mx = max(ns, default=1)
        ns_sc = [350 + (s/mx)*3500 for s in ns]
        ncolors = [MATLAB_COLORS[nc.get(n, 0) % len(MATLAB_COLORS)] for n in sG.nodes]
        ew    = [sG[u][v]['weight'] for u, v in sG.edges()]; me = max(ew, default=1)
        ew_sc = [0.3 + (w/me)*5   for w in ew]
        ea    = [0.06 + (w/me)*0.35 for w in ew]
        pos   = nx.spring_layout(sG, k=3.0, iterations=100, seed=42, weight='weight')
 
        for (u, v), wid, alp in zip(sG.edges(), ew_sc, ea):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='black', linewidth=wid, alpha=alp, zorder=1)
        for i, node in enumerate(sG.nodes):
            ax.scatter(pos[node][0], pos[node][1], s=ns_sc[i], c=ncolors[i],
                       edgecolors='black', linewidths=1.5, alpha=1.0, zorder=3)
        texts = []
        for i, node in enumerate(sG.nodes):
            lab = node.title()
            if len(lab) > 35: lab = lab[:32] + '…'
            fs = 20 + (kf.get(node, 0)/mx)*4.5
            t  = ax.text(pos[node][0], pos[node][1], lab,
                         fontsize=max(fs, 22), fontweight='bold',
                         ha='center', va='center', color='#1a1a1a', zorder=4)
            texts.append(t)
        try:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='#888888', alpha=0.4, lw=0.5),
                        expand_points=(2.0, 2.0), expand_text=(1.5, 1.5))
        except: pass
 
        # Source legend (top-left)
        src_labels = []
        if SOURCE_CONFIG['use_title']:           src_labels.append('Title n-grams')
        if SOURCE_CONFIG['use_abstract']:        src_labels.append('Abstract n-grams')
        if SOURCE_CONFIG['use_author_keywords']: src_labels.append('Author Keywords')
        if SOURCE_CONFIG['use_index_keywords']:  src_labels.append('Index Keywords')
        # ax.text(0.01, 0.99, 'Sources: ' + ', '.join(src_labels),
        #         transform=ax.transAxes, fontsize=11, va='top', ha='left',
        #         color='#444444', style='italic',
        #         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='#cccccc'))
 
        ax.axis('off'); ax.set_facecolor('#fafafa'); fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.close()
        print(f"  ✓ {os.path.basename(fp)}")
 
 
    # ── Individual plots ──────────────────────────────────────────────────────────
    for wl in all_results:
        tag = wl.replace('–', '-')
        plot_cooccur(all_results[wl], os.path.join(OUT, f'cooccurrence_{tag}0.4.png'))
        plot_cooccur(all_results[wl], os.path.join(OUT, f'cooccurrence_{tag}0.4.pdf'))
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 4-PANEL COMBINED PLOT
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\nPlotting combined 4-panel …")
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.subplots_adjust(hspace=0.22, wspace=0.15)
 
    for idx, (wl, _, _) in enumerate(WINDOWS):
        ax = axes.flat[idx]
        if wl not in all_results:
            ax.text(0.5, 0.5, f'{wl}\ninsufficient data',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off'); continue
 
        res = all_results[wl]; G = res['G']; kf = res['kw_freq']
        nc  = res['node_comm']
        top30 = [k for k, _ in kf.most_common(200) if k in G.nodes][:30]
        sG    = G.subgraph(top30).copy()
        sG.remove_nodes_from(list(nx.isolates(sG)))
        if len(sG.nodes) < 3:
            ax.text(0.5, 0.5, 'too few nodes', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.axis('off'); continue
 
        ns    = [kf.get(n, 10) for n in sG.nodes]; mx = max(ns, default=1)
        ns_sc = [150 + (s/mx)*2000 for s in ns]
        ncolors = [MATLAB_COLORS[nc.get(n, 0) % len(MATLAB_COLORS)] for n in sG.nodes]
        ew    = [sG[u][v]['weight'] for u, v in sG.edges()]; me = max(ew, default=1)
        ew_sc = [0.2 + (w/me)*3.5 for w in ew]
        ea_sc = [0.06 + (w/me)*0.3 for w in ew]
        pos   = nx.spring_layout(sG, k=2.5, iterations=80, seed=42, weight='weight')
 
        for (u, v), wid, alp in zip(sG.edges(), ew_sc, ea_sc):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='black', linewidth=wid, alpha=alp, zorder=1)
        for i, node in enumerate(sG.nodes):
            ax.scatter(pos[node][0], pos[node][1], s=ns_sc[i], c=ncolors[i],
                       edgecolors='black', linewidths=1.5, alpha=1.0, zorder=3)
        texts = []
        for i, node in enumerate(sG.nodes):
            lab = node.title()
            if len(lab) > 28: lab = lab[:25] + '…'
            fs = 13 + (kf.get(node, 0)/mx)*4
            t  = ax.text(pos[node][0], pos[node][1], lab,
                         fontsize=max(fs, 14), fontweight='bold',
                         ha='center', va='center', color='#1a1a1a', zorder=4)
            texts.append(t)
        try:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='#888888', alpha=0.4, lw=0.4),
                        expand_points=(1.8, 1.8), expand_text=(1.4, 1.4))
        except: pass
 
        ax.text(-0.02, 1.02, f'({chr(97+idx)})', fontsize=12, fontweight='bold',
                transform=ax.transAxes, ha='right', va='bottom')
        ax.set_title(f'{wl}  (n = {res["n_docs"]})', fontsize=10, fontweight='bold', pad=5)
        ax.axis('off'); ax.set_facecolor('#fafafa')
 
    fig.patch.set_facecolor('white')
    plt.savefig(os.path.join(OUT, 'cooccurrence_combined.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'cooccurrence_combined.pdf'), dpi=300)
    plt.close()
    print("  ✓ cooccurrence_combined.png / .pdf")
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # GLOBAL SUMMARY CSV  (all windows combined)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\nWriting global summary …")
    summary_rows = []
    for wl in [w[0] for w in WINDOWS]:
        if wl not in all_results: continue
        res = all_results[wl]; kf = res['kw_freq']
        for rank, (kw, cnt) in enumerate(kf.most_common(100), 1):
            summary_rows.append({'Window': wl, 'Rank': rank, 'Keyword': kw, 'Frequency': cnt})
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, 'keyword_summary_all_windows.csv'), index=False)
    print("  ✓ keyword_summary_all_windows.csv")
 
    print(f"\nAll done!  →  {OUT}/")
    print("\nFiles generated per window:")
    print("  • cooccurrence_<window>.png / .pdf      — network plot")
    print("  • keyword_freq_<window>.csv             — keyword frequencies + paper IDs")
    print("  • cooccurrence_top_<window>.csv         — top co-occurrence pairs + paper IDs")
    print("  • paper_keyword_map_<window>.csv        — per-paper keyword list")
    print("Combined:")
    print("  • cooccurrence_combined.png / .pdf")
    print("  • keyword_summary_all_windows.csv")


if __name__ == "__main__":
    main()
