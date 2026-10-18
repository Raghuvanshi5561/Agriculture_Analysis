# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:19:28 2026

"""

"""
Bibliometric Thematic Map & Co-occurrence — Time-Windowed (v2, optimized)
==========================================================================
Keywords are assembled from author keywords, index keywords, and title bi/trigrams.
Time windows: 2010-2013, 2014-2017, 2018-2021, 2022-2025
Climate-resilience cluster tracking.
"""
 
import pandas as pd
import numpy as np
import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from itertools import combinations
import networkx as nx
from networkx.algorithms.community import louvain_communities
from adjustText import adjust_text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run time-windowed bibliometric thematic mapping.")
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="Path to the cleaned/classified bibliographic CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/thematic_mapping"),
                        help="Directory for generated figures and CSV outputs (default: outputs/thematic_mapping).")
    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f"Input CSV does not exist: {args.input_csv}")
    OUT = str(args.output_dir)
    os.makedirs(OUT, exist_ok=True)

 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # ★ USER CONFIGURATION — CHANGE THESE PARAMETERS AS NEEDED ★
    # ═══════════════════════════════════════════════════════════════════════════════
    #
    # COMMUNITY_METHOD : 'louvain' or 'leiden'
    #   - 'louvain' : Blondel et al. 2008, widely cited in bibliometrics
    #   - 'leiden'  : Traag et al. 2019, guarantees connected communities (needs:
    #                 pip install leidenalg igraph)
    #
    # RESOLUTION : controls how many clusters you get
    #   - Higher → more, smaller clusters  |  Lower → fewer, larger clusters
    #   - For thematic maps, aim for 8–25 clusters
    #   - Set a single float (e.g. 1.5) to use the same value for ALL windows
    #   - Set None to use adaptive values that scale with window size (recommended)
    #
    # ADAPTIVE_RESOLUTION : only used when RESOLUTION = None
    #   - Dict mapping window size thresholds to resolution values
    #   - Format: {max_docs: resolution, ...}  — first matching threshold is used
    #
    COMMUNITY_METHOD = 'leiden'        # 'louvain' or 'leiden'
    RESOLUTION       = None             # float (e.g. 1.5) or None for adaptive
    ADAPTIVE_RESOLUTION = {             # used only when RESOLUTION = None
        800:   1.0,                     # windows < 800 docs
        3000:  1.2,                     # windows < 3000 docs
        8000:  1.5,                     # windows < 8000 docs
        99999: 2.0,                     # everything larger
    }
 
    # MIN_FREQ / EDGE_MIN per window size — controls network density
    # Format: {max_docs: (min_keyword_freq, min_edge_cooccurrence), ...}
    NETWORK_THRESHOLDS = {
        800:   (8,  3),
        3000:  (15, 5),
        8000:  (25, 8),
        99999: (40, 12),
    }
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. STOPWORDS & NORMALIZATION
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
        # single-word noise from abstracts
        'proposed method','proposed model','proposed system','proposed approach',
        'experimental results','results show','results showed','paper presents',
        'study aims','study investigates',
        # Publisher/boilerplate noise
        'rights reserved','elsevier right','elsevier rights reserved',
        'published elsevier','elsevier ltd','ltd right','ltd rights reserved',
        'open access','creativecommons org','creativecommons org licens',
        'access cc','license http creativecommon','http creativecommons org',
        'org licens','licens http','http creativecommon',
        # Generic filler phrases from abstracts
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
        # More publisher/license noise
        'open access cc','license http','cc nc','nc nd license','access cc nc',
        'nd license','license http creativecommon','licens http',
        # More filler phrases
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
        # More license/publisher boilerplate
        'cc nc nd','licenses nc nd','org licenses nc','nd license http',
        'elsevier behalf','cc nc','nc nd','nd license',
        # More generic filler
        'however existing','challenging due','training dataset',
        'enhance accuracy','accuracy achieved','accuracy score',
        'accuracy reliability','enhance efficiency','improving efficiency',
        'technological advancement','learning iot','learning predict',
        'learning ml','precision pa','driven decision','data driven decision',
        'crop classification','real health','health soil','plant monitoring',
        'plant health','crop health','informed decision',
        'resource allocation','identification',
    }
 
    NORM = {
        'agricultural':'agriculture','agric':'agriculture','agri':'agriculture',
        'smart agricultures':'smart agriculture',
        'modern agricultures':'smart agriculture',
        'machine-learning':'machine learning','deep-learning':'deep learning',
        'reinforcement-learning':'reinforcement learning',
        'transfer-learning':'transfer learning',
        'e-agriculture':'digital agriculture',
        'iot':'internet of things','ml':'machine learning',
        'ai':'artificial intelligence','dl':'deep learning',
        'cnn':'convolutional neural network','cnns':'convolutional neural network',
        'convolutional neural networks':'convolutional neural network',
        'ann':'artificial neural network','anns':'artificial neural network',
        'artificial neural networks':'artificial neural network',
        'uav':'unmanned aerial vehicle','uavs':'unmanned aerial vehicle',
        'unmanned aerial vehicles':'unmanned aerial vehicle',
        'unmanned aerial vehicles uav':'unmanned aerial vehicle',
        'unmanned aerial vehicles uavs':'unmanned aerial vehicle',
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
        'smart farming':'smart agriculture',
        'rs':'remote sensing',
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
        'plants botany':'botany',
        'classification of information':'classification',
        'learning algorithms':'learning algorithm',
        'learning systems':'learning system',
        'particle swarm optimization pso':'particle swarm optimization',
        'swarm optimization':'particle swarm optimization',
        'particle swarm':'particle swarm optimization',
        'genetic algorithms':'genetic algorithm',
        'decision trees':'decision tree',
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
        'objects detection':'object detection',
        'decisions makings':'decision making',
        'decision supports':'decision support system',
        'crop yield':'yield prediction','crop yield prediction':'yield prediction',
        'yield estimation':'yield prediction',
        'crop monitoring':'crop health monitoring',
        'image-analysis':'image analysis','image analyze':'image analysis',
        'climate resilient':'climate resilience',
        'climate-resilient':'climate resilience',
        'climate resilient agriculture':'climate resilient agriculture',
        'climate-resilient agriculture':'climate resilient agriculture',
        'climate smart':'climate smart agriculture',
        'climate-smart':'climate smart agriculture',
        'climate-smart agriculture':'climate smart agriculture',
        'drought resistant':'drought resistance',
        'drought-resistant':'drought resistance',
        'drought tolerant':'drought tolerance',
        'drought-tolerant':'drought tolerance',
        # Substring/partial duplicates
        'internet thing':'internet of things',
        'internet things':'internet of things',
        'things iot':'internet of things',
        'internet things iot':'internet of things',
        'internet of thing iot':'internet of things',
        'iot internet thing':'internet of things',
        'iot internet':'internet of things',
        'sensor networks':'sensor network',
        'sensor networks wsn':'wireless sensor network',
        'sensor network wsn':'wireless sensor network',
        'networks wireless sensor':'wireless sensor network',
        'sensor networks wireless':'wireless sensor network',
        'wireless sensor networks wsn':'wireless sensor network',
        'decision support systems':'decision support system',
        'decision support':'decision support system',
        'decision supports':'decision support system',
        'unmanned aerial':'unmanned aerial vehicle',
        'aerial vehicle':'unmanned aerial vehicle',
        'aerial vehicles':'unmanned aerial vehicle',
        'vehicles uav':'unmanned aerial vehicle',
        'vehicle uav':'unmanned aerial vehicle',
        'uav unmanned aerial':'unmanned aerial vehicle',
        'unmanned aerial vehicle uav':'unmanned aerial vehicle',
        'unmanned aircraft':'unmanned aerial vehicle',
        'unmanned aircraft uas':'unmanned aerial vehicle',
        'aircraft uas':'unmanned aerial vehicle',
        'robotic':'robotics',
        'agricultural robot':'agricultural robotics',
        'agricultural robots':'agricultural robotics',
        'intelligent robot':'agricultural robotics',
        'convolutional neural':'convolutional neural network',
        'network cnn':'convolutional neural network',
        'neural network cnn':'convolutional neural network',
        'deep learning dl':'deep learning',
        'learning dl':'deep learning',
        'learning deep':'deep learning',
        'learning approache':'deep learning',
        'learning approaches':'deep learning',
        'machine learning ml':'machine learning',
        'learning machine':'machine learning',
        'learning model':'machine learning',
        'learning algorithm':'machine learning algorithm',
        'learning smart':'smart agriculture',
        'learning system':'machine learning',
        'learning driven':'machine learning',
        'intelligence ai':'artificial intelligence',
        'artificial intelligence ai':'artificial intelligence',
        'ai driven':'artificial intelligence',
        'support vector':'support vector machine',
        'vector machine':'support vector machine',
        'difference vegetation':'vegetation index',
        'difference vegetation index':'vegetation index',
        'normalized difference vegetation':'vegetation index',
        'normalized difference vegetation index':'vegetation index',
        'moisture sensor':'soil moisture sensor',
        'soil moisture sensor':'soil moisture',
        'iot sensor':'internet of things',
        'term memory':'long short term memory',
        'images processing':'image processing',
        'pre processing':'preprocessing',
        'error rmse':'root mean square error',
        'mean square':'mean square error',
        'square error':'mean square error',
        'root mean':'root mean square error',
        'precision recall':'classification metrics',
        'accuracy rate':'accuracy',
        'higher accuracy':'accuracy',
        'feature extraction':'feature extraction',
        'network architecture':'neural network architecture',
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
 
    def _lemma_phrase(phrase):
        """Simple plural→singular for the LAST word of a multi-word phrase."""
        words = phrase.split()
        if not words:
            return phrase
        last = words[-1]
        # Don't lemmatize very short words or known exceptions
        if len(last) <= 3 or last in {'analysis', 'basis', 'thesis', 'diagnosis',
                                       'stress', 'loss', 'biomass', 'access',
                                       'process', 'progress', 'wireless', 'less',
                                       'across', 'class', 'grass', 'gas', 'focus'}:
            return phrase
        if last.endswith('ies') and len(last) > 4:
            words[-1] = last[:-3] + 'y'
        elif last.endswith('ses') or last.endswith('zes') or last.endswith('xes'):
            words[-1] = last[:-2]
        elif last.endswith('ves') and len(last) > 4:
            words[-1] = last[:-3] + 'f'  # leaves→leaf, halves→half
        elif last.endswith('s') and not last.endswith('ss') and not last.endswith('us'):
            words[-1] = last[:-1]
        return ' '.join(words)
 
 
    def clean_kw(kw):
        kw = kw.strip().lower()
        kw = re.sub(r'[^\w\s\-/]', '', kw)
        kw = re.sub(r'\s+', ' ', kw).strip()
        # Replace hyphens with spaces for consistency
        kw = kw.replace('-', ' ').strip()
        kw = re.sub(r'\s+', ' ', kw).strip()
        # Manual normalization first
        if kw in NORM: kw = NORM[kw]
        # Then lemmatize plurals
        kw = _lemma_phrase(kw)
        # Re-check NORM after lemmatization
        if kw in NORM: kw = NORM[kw]
        return kw
 
 
    def is_valid(kw):
        return (kw and len(kw) > 2
                and kw not in stop_words
                and kw not in OBVIOUS
                and not all(c in '- /' for c in kw)
                and not kw.isdigit())
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. LOAD DATA
    # ═══════════════════════════════════════════════════════════════════════════════
    print("Loading data …")
    df = pd.read_csv(args.input_csv, low_memory=False)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    print(f"  Records: {len(df)}, Years: {df['Year'].min():.0f}–{df['Year'].max():.0f}")
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. EXTRACT TITLE BI/TRIGRAMS (no unigrams — too noisy)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("Extracting title+abstract bi/trigrams …")
    #df['text'] = (df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')).str.lower()
    df['text'] = (df['Title'].fillna('')).str.lower()
    df['text'] = df['text'].apply(lambda t: re.sub(r'[^a-z\s]', ' ', t))
    df['text'] = df['text'].apply(lambda t: re.sub(r'\s+', ' ', t).strip())
 
    extra_stop = list(stop_words | OBVIOUS)
    vec = CountVectorizer(ngram_range=(2, 3), min_df=50, max_df=0.5,
                          stop_words=extra_stop, max_features=3000)
    X = vec.fit_transform(df['text'])
    vocab = vec.get_feature_names_out()
    print(f"  Text-derived phrases (min_df≥50): {len(vocab)}")
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. MERGE ALL KEYWORDS
    # ═══════════════════════════════════════════════════════════════════════════════
    print("Merging all keyword sources …")
 
    def extract_all_kw(idx, row):
        """Extract normalized keywords from author/index fields and title n-grams."""
        kws = set()
        for col in ('Author Keywords', 'Index Keywords'):
            val = row.get(col, '')
            if pd.notna(val) and str(val).strip():
                for phrase in str(val).split(';'):
                    cleaned = clean_kw(phrase)
                    if is_valid(cleaned):
                        kws.add(cleaned)
        for col_idx in X[idx].nonzero()[1]:
            cleaned = clean_kw(vocab[col_idx])
            if is_valid(cleaned):
                kws.add(cleaned)
        return kws

    all_kw_lists = [extract_all_kw(i, df.iloc[i]) for i in range(len(df))]
    df['_kws'] = all_kw_lists
    df['all_keywords'] = all_kw_lists

    flat_global = [k for ks in df['_kws'] for k in ks]
    kw_freq_global = Counter(flat_global)
    print(f"  Unique keywords (before dedup): {len(kw_freq_global)}")
    # Build global keyword → paper-index map for merge validation
    kw_to_papers_global = defaultdict(set)
    for i, ks in enumerate(df['_kws']):
        for k in ks:
            kw_to_papers_global[k].add(i)
    # ═══════════════════════════════════════════════════════════════════════════════
    # 4b. AUTOMATED SUBSTRING & NEAR-DUPLICATE DEDUPLICATION
    # ═══════════════════════════════════════════════════════════════════════════════
    print("Running automated deduplication …")
 
    # def build_auto_merge(freq, min_count=20):
    #     """
    #     Build a merge dictionary automatically:
    #       1. Substring merging:  'internet things' → 'internet of things'
    #          (shorter phrase is a subset of words in a longer, more frequent phrase)
    #       2. Stopword-only difference: 'decision support' vs 'decision support system'
    #          (one is the other plus a stopword or a common suffix)
    #       3. Partial overlap: 'unmanned aerial' → 'unmanned aerial vehicle'
    #          (one is a strict prefix/suffix of the other)
    #     """
    #     # Only consider keywords with at least min_count occurrences
    #     candidates = {k for k, c in freq.items() if c >= min_count and ' ' in k}
    #     merge = {}
 
    #     # Sort by length descending so we merge short→long
    #     by_len = sorted(candidates, key=lambda x: (-len(x.split()), -freq[x]))
 
    #     # Build word-set index for fast lookup
    #     word_sets = {k: set(k.split()) for k in candidates}
 
    #     for i, short in enumerate(by_len):
    #         if short in merge:  # already merged
    #             continue
    #         if short in PROTECT: continue
    #         s_words = word_sets[short]
    #         s_freq = freq[short]
 
    #         best_target = None
    #         best_freq = 0
 
    #         for long in by_len:
    #             if long == short or long in merge:
    #                 continue
    #             l_words = word_sets[long]
 
    #             # Skip if same word count — not a substring relationship
    #             if len(l_words) <= len(s_words):
    #                 continue
 
    #             # Check: all words of 'short' appear in 'long' (ignoring stopwords)
    #             diff = l_words - s_words
    #             # The difference should be small (1-2 words) and ideally stopwords
    #             # or very common suffixes
    #             if s_words.issubset(l_words) and len(diff) <= 2:
    #                 l_freq = freq[long]
    #                 # Merge the LESS frequent into the MORE frequent
    #                 if l_freq > best_freq:
    #                     best_target = long
    #                     best_freq = l_freq
 
    #         if best_target and best_freq >= s_freq * 0.3:
    #             # Merge short → long (the more complete form)
    #             if best_freq >= s_freq:
    #                 merge[short] = best_target
    #             else:
    #                 merge[best_target] = short
 
    #     # Also catch exact-prefix duplicates not caught above
    #     # e.g., 'sensor network' vs 'sensor networks' (after lemmatization these should
    #     # be the same, but in case they slipped through)
    #     all_kws = sorted(freq.keys(), key=lambda x: (-freq[x],))
    #     for kw in all_kws:
    #         if kw in merge:
    #             continue
    #         # Check if adding/removing trailing 's' gives a more frequent form
    #         for variant in [kw + 's', kw + 'es']:
    #             if variant in freq and variant not in merge and variant != kw:
    #                 if freq[variant] < freq[kw]:
    #                     merge[variant] = kw
    #                 else:
    #                     merge[kw] = variant
    #                     break
 
    #     return merge
 
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





    AUTO_MERGE = build_auto_merge(kw_freq_global, kw_to_papers_global, min_count=10, min_jaccard=0.4)
    def apply_merge(kw_set, merge_dict):
        return {merge_dict.get(k, k) for k in kw_set}
 
    df['_kws'] = df['_kws'].apply(lambda ks: apply_merge(ks, AUTO_MERGE))
    df['all_keywords'] = df['_kws']
    flat_global = [k for ks in df['_kws'] for k in ks]
    kw_freq_global = Counter(flat_global)
    print(f"  Unique keywords (after dedup): {len(kw_freq_global)}")
 


    # AUTO_MERGE = build_auto_merge(kw_freq_global, min_count=10)
    # print(f"  Auto-merge rules: {len(AUTO_MERGE)}")
    # # Print a sample of merges
    # sample = list(AUTO_MERGE.items())[:20]
    # for src, tgt in sample:
    #     print(f"    '{src}' → '{tgt}'  ({kw_freq_global[src]}→{kw_freq_global[tgt]})")
    # if len(AUTO_MERGE) > 20:
    #     print(f"    ... and {len(AUTO_MERGE)-20} more")
 
    # # Apply auto-merge to all keyword lists
    # def apply_merge(kw_list, merge_dict):
    #     merged = set()
    #     for k in kw_list:
    #         merged.add(merge_dict.get(k, k))
    #     return list(merged)
 
    # df['all_keywords'] = df['all_keywords'].apply(lambda ks: apply_merge(ks, AUTO_MERGE))
 
    # # Recount
    # flat = [k for ks in df['all_keywords'] for k in ks]
    # kw_freq_global = Counter(flat)
    # print(f"  Unique keywords (after dedup): {len(kw_freq_global)}")
    # print(f"  Top 15: {kw_freq_global.most_common(15)}")
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 5. ANALYSIS & PLOTTING
    # ═══════════════════════════════════════════════════════════════════════════════
    WINDOWS = [
        ('2010–2013', 2010, 2013),
        ('2014–2017', 2014, 2017),
        ('2018–2021', 2018, 2021),
        ('2022–2025', 2022, 2025),
    ]
 
    # ── MATLAB-style vibrant palette ──
    QCOLORS = {
        'Motor Themes':                   '#D62728',   # MATLAB red
        'Basic Themes':                   '#1F77B4',   # MATLAB blue
        'Niche Themes':                   '#FF7F0E',   # MATLAB orange
        'Emerging or\nDeclining Themes':  '#2CA02C',   # MATLAB green
    }
 
 
    def run_window(sub_df, label, min_freq, edge_min, resolution):
        print(f"\n{'='*60}\n  {label}  ({len(sub_df)} docs)\n{'='*60}")
        kw_lists = sub_df['all_keywords'].tolist()
        flat = [k for ks in kw_lists for k in ks]
        kw_freq = Counter(flat)
 
        freq_kws = {k for k, c in kw_freq.items() if c >= min_freq}
        print(f"  Frequent keywords (≥{min_freq}): {len(freq_kws)}")
        if len(freq_kws) < 15:
            print("  ⚠ Too few, skipping"); return None
 
        cooccur = defaultdict(int)
        for ks in kw_lists:
            filt = sorted(set(k for k in ks if k in freq_kws))
            for a, b in combinations(filt, 2):
                cooccur[(a, b)] += 1
 
        G = nx.Graph()
        for (a, b), w in cooccur.items():
            if w >= edge_min:
                G.add_edge(a, b, weight=w)
        if len(G.nodes) < 10:
            print("  ⚠ Network too small"); return None
 
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        print(f"  Network: {len(G.nodes)} nodes, {len(G.edges)} edges")
 
        # Cap for performance: if still too large, raise threshold
        if len(G.edges) > 100000:
            # Re-filter edges to top by weight
            edge_threshold = sorted([d['weight'] for _, _, d in G.edges(data=True)],
                                    reverse=True)[min(80000, len(G.edges)-1)]
            G2 = nx.Graph()
            for u, v, d in G.edges(data=True):
                if d['weight'] >= edge_threshold:
                    G2.add_edge(u, v, weight=d['weight'])
            largest = max(nx.connected_components(G2), key=len)
            G = G2.subgraph(largest).copy()
            print(f"  Pruned to: {len(G.nodes)} nodes, {len(G.edges)} edges")
 
        # ── Community detection ──
        print(f"  Method: {COMMUNITY_METHOD}, resolution={resolution}")
        if COMMUNITY_METHOD == 'leiden':
            import igraph as ig
            import leidenalg
            # Convert NetworkX → igraph
            nodes_list = list(G.nodes())
            node_idx = {n: i for i, n in enumerate(nodes_list)}
            edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            G_ig = ig.Graph(n=len(nodes_list), edges=edges, directed=False)
            G_ig.vs['name'] = nodes_list
            G_ig.es['weight'] = weights
            part = leidenalg.find_partition(
                G_ig, leidenalg.RBConfigurationVertexPartition,
                weights='weight', resolution_parameter=resolution, seed=42
            )
            comms = [set(G_ig.vs[m]['name'] for m in members) for members in part]
        else:  # louvain (default)
            comms = list(louvain_communities(G, weight='weight',
                                             resolution=resolution, seed=42))
 
        # Retry with higher resolution if too few clusters
        if len([c for c in comms if len(c) >= 3]) < 4:
            retry_res = resolution * 1.8
            print(f"  Too few clusters, retrying with resolution={retry_res:.1f}")
            if COMMUNITY_METHOD == 'leiden':
                part = leidenalg.find_partition(
                    G_ig, leidenalg.RBConfigurationVertexPartition,
                    weights='weight', resolution_parameter=retry_res, seed=42
                )
                comms = [set(G_ig.vs[m]['name'] for m in members) for members in part]
            else:
                comms = list(louvain_communities(G, weight='weight',
                                                 resolution=retry_res, seed=42))
 
        comms = [c for c in comms if len(c) >= 3]
        print(f"  Communities (≥3): {len(comms)}")
 
        node_comm = {}
        for i, c in enumerate(comms):
            for n in c: node_comm[n] = i
 
        cluster_data = []
        for i, comm in enumerate(comms):
            nodes = list(comm)
            if len(nodes) < 3: continue
            sG = G.subgraph(nodes)
            iw = [d['weight'] for _, _, d in sG.edges(data=True)]
            density = np.mean(iw) if iw else 0
            ext_w = sum(G[n][nb]['weight'] for n in nodes for nb in G.neighbors(n) if nb not in comm)
            centrality = ext_w / len(nodes)
 
            nf = {n: kw_freq.get(n, 0) for n in nodes}
            top3 = sorted(nf, key=nf.get, reverse=True)[:3]
 
            cluster_data.append({
                'id': i, 'centrality': centrality, 'density': density,
                'label': '\n'.join(k.title() for k in top3),
                'size': len(nodes), 'total_freq': sum(nf.values()),
                'top_keywords': top3, 'all_keywords': nodes,
            })
 
        cdf = pd.DataFrame(cluster_data)
        if len(cdf) == 0: return None
 
        cm = cdf['centrality'].median()
        dm = cdf['density'].median()
 
        def quad(r):
            hc, hd = r['centrality']>=cm, r['density']>=dm
            if hc and hd: return 'Motor Themes'
            if hc and not hd: return 'Basic Themes'
            if not hc and hd: return 'Niche Themes'
            return 'Emerging or\nDeclining Themes'
 
        cdf['quadrant'] = cdf.apply(quad, axis=1)
 
        for _, r in cdf.sort_values('quadrant').iterrows():
            q = r['quadrant'].replace('\n',' ')
            top5 = sorted(r['all_keywords'], key=lambda k: kw_freq.get(k,0), reverse=True)[:5]
            print(f"    [{q}] C={r['centrality']:.0f} D={r['density']:.1f} → {', '.join(top5)}")
 
        return {'G': G, 'comms': comms, 'node_comm': node_comm, 'cdf': cdf,
                'kw_freq': kw_freq, 'cent_med': cm, 'dens_med': dm,
                'window': label, 'n_docs': len(sub_df)}
 
 
    all_results = {}
    for wl, ys, ye in WINDOWS:
        sub = df[(df['Year']>=ys) & (df['Year']<=ye)]
        if len(sub) < 30: continue
 
        n = len(sub)
 
        # Pick network thresholds from config
        for threshold, (mf, em) in sorted(NETWORK_THRESHOLDS.items()):
            if n < threshold:
                break
 
        # Pick resolution: fixed value or adaptive from config
        if RESOLUTION is not None:
            rv = RESOLUTION
        else:
            for threshold, rv in sorted(ADAPTIVE_RESOLUTION.items()):
                if n < threshold:
                    break
 
        r = run_window(sub, wl, mf, em, rv)
        if r: all_results[wl] = r
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 6. INDIVIDUAL PLOTS (standalone, high-resolution)
    # ═══════════════════════════════════════════════════════════════════════════════
 
    # MATLAB-style vibrant colormap for co-occurrence clusters
    MATLAB_COLORS = [
        '#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
        '#4DBEEE', '#A2142F', '#FF6F61', '#6B5B95', '#88B04B',
        '#F7CAC9', '#92A8D1', '#955251', '#B565A7', '#009B77',
        '#DD4124', '#45B8AC', '#5B5EA6', '#E15D44', '#9B2335',
    ]
 
 
    def plot_thematic(res, fp):
        """Individual thematic map — journal-ready at 300 dpi."""
        cdf = res['cdf']
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axhline(res['dens_med'], color='#555555', ls='--', lw=1.0, alpha=0.5)
        ax.axvline(res['cent_med'], color='#555555', ls='--', lw=1.0, alpha=0.5)
 
        # Axis limits with generous padding
        xmin, xmax = cdf['centrality'].min(), cdf['centrality'].max()
        ymin, ymax = cdf['density'].min(), cdf['density'].max()
        xpad = (xmax - xmin) * 0.15
        ypad = (ymax - ymin) * 0.15
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
 
        # Quadrant labels — fixed at corners via transAxes
        for tag, (xa, ya, ha, va) in {
            'MOTOR THEMES':        (0.98, 0.98, 'right', 'top'),
            'NICHE THEMES':        (0.02, 0.98, 'left',  'top'),
            'BASIC THEMES':        (0.98, 0.02, 'right', 'bottom'),
            'EMERGING / DECLINING':(0.02, 0.02, 'left',  'bottom'),
        }.items():
            ax.text(xa, ya, tag, ha=ha, va=va, fontsize=14, fontstyle='italic',
                    color='#888888', fontweight='bold', alpha=0.6,
                    transform=ax.transAxes)
 
        mf = max(cdf['total_freq'].max(), 1)
        texts = []
        for _, r in cdf.iterrows():
            c = QCOLORS[r['quadrant']]
            sz = 500 + (r['total_freq']/mf)*3000
            ax.scatter(r['centrality'], r['density'], s=sz, c=c, alpha=0.55,
                       edgecolors='white', linewidths=1.5, zorder=5)
            t = ax.text(r['centrality'], r['density'], r['label'],
                        ha='center', va='center', fontsize=16,
                        fontweight='bold', color='#1a1a1a', zorder=6)
            texts.append(t)
        try:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='#666666', alpha=0.5, lw=0.8),
                        expand_points=(2.5, 2.5), expand_text=(1.8, 1.8),
                        force_points=(1.2, 1.2), force_text=(1.0, 1.0))
        except: pass
 
        ax.set_xlabel('Centrality (Relevance degree)', fontsize=22, fontweight='bold')
        ax.set_ylabel('Density (Development degree)', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True, alpha=0.12, color='#cccccc')
        ax.set_facecolor('#fafafa'); fig.patch.set_facecolor('white')
 
        # h = [mpatches.Patch(fc=v, alpha=0.65, label=k.replace('\n',' ')) for k,v in QCOLORS.items()]
        # ax.legend(handles=h, loc='upper left', framealpha=0.9, fontsize=10,
        #           edgecolor='#cccccc', fancybox=False)
        plt.tight_layout()
        plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.close()
        print(f"  ✓ {os.path.basename(fp)}")
 
 
    def plot_cooccur(res, fp, top_n=50):
        """Individual co-occurrence network — journal-ready at 300 dpi."""
        G = res['G']; kf = res['kw_freq']; comms = res['comms']; nc = res['node_comm']
        top = [k for k,_ in kf.most_common(300) if k in G.nodes][:top_n]
        sG = G.subgraph(top).copy()
        sG.remove_nodes_from(list(nx.isolates(sG)))
        if len(sG.nodes) < 3: return
        print(f"  Co-occ viz: {len(sG.nodes)} nodes, {len(sG.edges)} edges")
 
        fig, ax = plt.subplots(figsize=(18, 15))
        ns = [kf.get(n,12) for n in sG.nodes]; mx = max(ns, default=1)
        ns_sc = [350+(s/mx)*3500 for s in ns]
        ncolors = [MATLAB_COLORS[nc.get(n,0) % len(MATLAB_COLORS)] for n in sG.nodes]
        ew = [sG[u][v]['weight'] for u,v in sG.edges()]; me = max(ew, default=1)
        ew_sc = [0.3+(w/me)*5 for w in ew]
        ea = [0.06+(w/me)*0.35 for w in ew]
        pos = nx.spring_layout(sG, k=3.0, iterations=100, seed=42, weight='weight')
 
        for (u,v),wid,alp in zip(sG.edges(), ew_sc, ea):
            ax.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]],
                    color='black', linewidth=wid, alpha=alp, zorder=1)
        for i,node in enumerate(sG.nodes):
            # ax.scatter(pos[node][0], pos[node][1], s=ns_sc[i], c=ncolors[i],
            #            edgecolors='#2c2c2c', linewidths=0.8, alpha=0.85, zorder=3)
            ax.scatter(pos[node][0], pos[node][1], s=ns_sc[i], c=ncolors[i],
        edgecolors='black', linewidths=1.5, alpha=1.0, zorder=3)
        texts = []
        for i,node in enumerate(sG.nodes):
            lab = node.title()
            if len(lab)>35: lab = lab[:32]+'…'
            fs = 15+(kf.get(node,0)/mx)*4.5
            t = ax.text(pos[node][0], pos[node][1], lab,
                        fontsize=max(fs,16), fontweight='bold',
                        ha='center', va='center', color='#1a1a1a', zorder=4)
            texts.append(t)
        try:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='#888888', alpha=0.4, lw=0.5),
                        expand_points=(2.0,2.0), expand_text=(1.5,1.5))
        except: pass
 
        ax.axis('off'); ax.set_facecolor('#fafafa'); fig.patch.set_facecolor('white')
        for sl,sv in [('Low freq',500),('Medium',1800),('High freq',3800)]:
            ax.scatter([],[],s=sv,c='#cccccc',edgecolors='#2c2c2c',linewidths=0.8,label=sl)
        # ax.legend(title='Keyword Frequency', loc='lower left', framealpha=0.9,
        #           fontsize=9, title_fontsize=10, edgecolor='#cccccc', fancybox=False)
        plt.tight_layout()
        plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.close()
        print(f"  ✓ {os.path.basename(fp)}")
 
 
    # Generate individual plots
    for wl in all_results:
        tag = wl.replace('–','-')
        plot_thematic(all_results[wl], os.path.join(OUT, f'thematic_map_{tag}F.png'))
        plot_cooccur(all_results[wl], os.path.join(OUT, f'cooccurrence_{tag}.png'))
        plot_thematic(all_results[wl], os.path.join(OUT, f'thematic_map_{tag}F.pdf'))
        plot_cooccur(all_results[wl], os.path.join(OUT, f'cooccurrence_{tag}.pdf'))
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 7. COMBINED 4-PANEL PLOTS — A4-page sized for double-column LaTeX journal
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\nPlotting combined panels …")
 
    # ── 4-panel Thematic Map  (A4 landscape-ish, ~190mm wide at 300 dpi) ──
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)
 
    for idx, (wl, _, _) in enumerate(WINDOWS):
        ax = axes.flat[idx]
        if wl not in all_results:
            ax.text(0.5, 0.5, f'{wl}\n(insufficient data)', ha='center', va='center',
                    fontsize=11, transform=ax.transAxes)
            ax.set_facecolor('#fafafa'); continue
        res = all_results[wl]; cdf = res['cdf']
        ax.axhline(res['dens_med'], color='#555555', ls='--', lw=0.7, alpha=0.5)
        ax.axvline(res['cent_med'], color='#555555', ls='--', lw=0.7, alpha=0.5)
 
        # Axis limits with padding
        xmin, xmax = cdf['centrality'].min(), cdf['centrality'].max()
        ymin, ymax = cdf['density'].min(), cdf['density'].max()
        xpad = (xmax - xmin) * 0.15
        ypad = (ymax - ymin) * 0.15
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
 
        # Quadrant labels — corners via transAxes
        for tag, (xa, ya, ha, va) in {
            'MOTOR':    (0.97, 0.97, 'right', 'top'),
            'NICHE':    (0.03, 0.97, 'left',  'top'),
            'BASIC':    (0.97, 0.03, 'right', 'bottom'),
            'EMERGING': (0.03, 0.03, 'left',  'bottom'),
        }.items():
            ax.text(xa, ya, tag, ha=ha, va=va, fontsize=12, fontstyle='italic',
                    color='#888888', fontweight='bold', alpha=0.6,
                    transform=ax.transAxes)
 
        mf = max(cdf['total_freq'].max(), 1)
        texts = []
        for _, r in cdf.iterrows():
            c = QCOLORS[r['quadrant']]
            sz = 180 + (r['total_freq']/mf)*1500
            ax.scatter(r['centrality'], r['density'], s=sz, c=c, alpha=0.55,
                       edgecolors='white', linewidths=1.0, zorder=5)
            t = ax.text(r['centrality'], r['density'], r['label'],
                        ha='center', va='center', fontsize=12,
                        fontweight='bold', color='#1a1a1a', zorder=6)
            texts.append(t)
        try:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='#666666', alpha=0.4, lw=0.5),
                        expand_points=(2.2, 2.2), expand_text=(1.6, 1.6),
                        force_points=(0.9, 0.9), force_text=(0.7, 0.7))
        except: pass
 
        ax.set_xlabel('Centrality', fontsize=14, fontweight='bold')
        ax.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.text(-0.08, 1.02, f'({chr(97+idx)})', fontsize=16, fontweight='bold',
                transform=ax.transAxes, ha='right', va='bottom')
        # ax.set_title(f'{wl}  (n = {res["n_docs"]})',
        #              fontsize=10, fontweight='bold', pad=6)
        ax.grid(True, alpha=0.10, color='#cccccc')
        ax.set_facecolor('#fafafa')
 
    # h = [mpatches.Patch(fc=v, alpha=0.65, label=k.replace('\n',' ')) for k,v in QCOLORS.items()]
    # fig.legend(handles=h, loc='lower center', ncol=4, framealpha=0.9, fontsize=9,
    #            edgecolor='#cccccc', fancybox=False, bbox_to_anchor=(0.5, -0.01))
    # fig.patch.set_facecolor('white')
    plt.savefig(os.path.join(OUT, 'thematic_map_combined.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'thematic_map_combined.pdf'), dpi=300)
    plt.close()
    print("  ✓ thematic_map_combined.png")
 
 
    # ── 4-panel Co-occurrence Network (A4 landscape-ish) ──
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.subplots_adjust(hspace=0.22, wspace=0.15)
 
    for idx, (wl, _, _) in enumerate(WINDOWS):
        ax = axes.flat[idx]
        if wl not in all_results:
            ax.text(0.5, 0.5, f'{wl}\ninsufficient data', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.axis('off'); continue
        res = all_results[wl]; G = res['G']; kf = res['kw_freq']
        comms = res['comms']; nc = res['node_comm']
        top30 = [k for k,_ in kf.most_common(200) if k in G.nodes][:30]
        sG = G.subgraph(top30).copy()
        sG.remove_nodes_from(list(nx.isolates(sG)))
        if len(sG.nodes) < 3:
            ax.text(0.5, 0.5, 'too few nodes', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.axis('off'); continue
 
        ns = [kf.get(n,10) for n in sG.nodes]; mx = max(ns, default=1)
        ns_sc = [150+(s/mx)*2000 for s in ns]
        ncolors = [MATLAB_COLORS[nc.get(n,0) % len(MATLAB_COLORS)] for n in sG.nodes]
        ew = [sG[u][v]['weight'] for u,v in sG.edges()]; me = max(ew, default=1)
        ew_sc = [0.2+(w/me)*3.5 for w in ew]
        ea_sc = [0.06+(w/me)*0.3 for w in ew]
        pos = nx.spring_layout(sG, k=2.5, iterations=80, seed=42, weight='weight')
 
        for (u,v),wid,alp in zip(sG.edges(), ew_sc, ea_sc):
            ax.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]],
                    color='black', linewidth=wid, alpha=alp, zorder=1)
        for i, node in enumerate(sG.nodes):
            # ax.scatter(pos[node][0], pos[node][1], s=ns_sc[i], c=ncolors[i],
            #            edgecolors='#2c2c2c', linewidths=0.6, alpha=0.85, zorder=3)
            ax.scatter(pos[node][0], pos[node][1], s=ns_sc[i], c=ncolors[i], edgecolors='black',
        linewidths=1.5,alpha=1.0, zorder=3)
        texts = []
        for i, node in enumerate(sG.nodes):
            lab = node.title()
            if len(lab) > 28: lab = lab[:25]+'…'
            fs = 13+(kf.get(node,0)/mx)*4
            t = ax.text(pos[node][0], pos[node][1], lab,
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
        # ax.set_title(f'{wl}  (n = {res["n_docs"]})',
        #              fontsize=10, fontweight='bold', pad=5)
        ax.axis('off'); ax.set_facecolor('#fafafa')
 
    fig.patch.set_facecolor('white')
    plt.savefig(os.path.join(OUT, 'cooccurrence_combined.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'cooccurrence_combined.pdf'), dpi=300)
    plt.close()
    print("  ✓ cooccurrence_combined.png")
 
 
    # ═══════════════════════════════════════════════════════════════════════════════
    # 8. SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("CLUSTER SUMMARY")
    print("="*70)
    for wl in [w[0] for w in WINDOWS]:
        if wl not in all_results: continue
        cdf = all_results[wl]['cdf']; kf = all_results[wl]['kw_freq']
        print(f"\n  {wl}:")
        for _, r in cdf.sort_values('quadrant').iterrows():
            q = r['quadrant'].replace('\n',' ')
            top5 = sorted(r['all_keywords'], key=lambda k: kf.get(k,0), reverse=True)[:5]
            print(f"    [{q}] C={r['centrality']:.0f} D={r['density']:.1f} → {', '.join(top5)}")
 
    print(f"\n\nAll done! Output → {OUT}/")


if __name__ == "__main__":
    main()
