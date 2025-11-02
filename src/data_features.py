# features_extraction.py
# ------------------------------------------------------------
# Extraction des features à partir de train_clean.csv :
# - TF-IDF sur LISTES d'actions (unigram + bigram) via analyzer callable (niveau module, picklable)
# - Topics SVD(20) sur TF-IDF
# - Features séquentielles / temporelles / ratios sémantiques
# - Divisions protégées (pas de RuntimeWarning)
# - Artefacts sérialisés (TF-IDF, SVD, vocabs...)
# Sortie : train_features_ready.csv
# ------------------------------------------------------------

import os
import re
import math
import json
import logging
from dataclasses import dataclass, asdict
from collections import Counter
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ===================== PARAMS =====================
TRANSFORM_ONLY = False
PROJECT_ROOT = Path.cwd()
INPUT_FILE = PROJECT_ROOT / "data" / "train_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "train_features_ready.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts_vscode"

SECONDS_PER_TICK = 5
TOP_K_ACTIONS = 36
TFIDF_MAX_FEATURES = 400
TFIDF_NGRAM_MAX = 2
MIN_NAV_CAT_FREQ_RATIO = 0.01
ADD_NAV_INTERACTIONS = True
TOP_K_ACTION_BIGRAMS = 20
SVD_N_COMPONENTS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("features")

# ----------------------- Helpers -----------------------
def nettoie_action(token: str) -> str:
    if not isinstance(token, str) or token == "":
        return ""
    for sep in ["(", "<", "$", "1"]:
        if sep in token:
            i = token.index(sep)
            if i > 0:
                token = token[:i]
    token = re.sub(r"\s+", " ", token).strip()
    return token

def decoupe_trace(trace: str) -> List[str]:
    if not isinstance(trace, str) or trace == "":
        return []
    return [t.strip() for t in trace.split(",") if t.strip()]

def actions_nettoyees_sans_t(trace_str: str) -> List[str]:
    acts = []
    for tok in decoupe_trace(trace_str):
        if tok.startswith("t"):
            continue
        a = nettoie_action(tok)
        if a:
            acts.append(a)
    return acts

def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return float(ent)

def top2_ratio_from_counts(cnt: Counter, n_total: int) -> float:
    if n_total <= 0 or not cnt:
        return 0.0
    top2 = sorted(cnt.values(), reverse=True)[:2]
    return float(sum(top2) / n_total)

def gini_from_counts(cnt: Counter) -> float:
    total = sum(cnt.values())
    if total <= 0:
        return 0.0
    p2 = sum((c / total) ** 2 for c in cnt.values())
    return float(1.0 - p2)

def simpson_from_counts(cnt: Counter) -> float:
    total = sum(cnt.values())
    if total <= 0:
        return 0.0
    return float(sum((c / total) ** 2 for c in cnt.values()))

def plus_frequent(pattern: re.Pattern, texte: str) -> Optional[str]:
    if not isinstance(texte, str) or not texte:
        return None
    matches = pattern.findall(texte)
    return Counter(matches).most_common(1)[0][0] if matches else None

def count_repeats(actions: List[str]) -> int:
    """Nombre de répétitions consécutives (runs-1 cumulés)."""
    if not actions:
        return 0
    repeats = 0
    prev = actions[0]
    run = 1
    for a in actions[1:]:
        if a == prev:
            run += 1
        else:
            if run > 1:
                repeats += (run - 1)
            prev = a
            run = 1
    if run > 1:
        repeats += (run - 1)
    return repeats

def actions_per_tick(trace_str: str) -> List[int]:
    """Compte nb d'actions (hors t*) entre chaque tN."""
    toks = decoupe_trace(trace_str)
    counts = []
    c = 0
    saw_any_t = False
    for tok in toks:
        if tok.startswith("t"):
            saw_any_t = True
            counts.append(c)
            c = 0
        else:
            if nettoie_action(tok):
                c += 1
    if not saw_any_t:
        return [c] if (c > 0 or len(toks) > 0) else []
    counts.append(c)
    return counts

def bigrams_from_actions(actions: List[str]) -> List[str]:
    if len(actions) < 2:
        return []
    return [f"{actions[i]}|||{actions[i+1]}" for i in range(len(actions) - 1)]

# === Analyzer callable picklable (NIVEAU MODULE) ===
def ngram_actions(tokens: List[str]) -> List[str]:
    """
    Reçoit une LISTE d'actions ; renvoie unigrams + bigrams (séparateur '|||').
    Définie au niveau module => picklable par joblib.
    """
    if not isinstance(tokens, list):
        return []
    if len(tokens) < 2:
        return tokens[:]
    bigrams = [f"{tokens[i]}|||{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams

# ----------------------- Config & Artefacts -----------------------
@dataclass
class FeatureConfig:
    seconds_per_tick: int = SECONDS_PER_TICK
    top_k_actions: int = TOP_K_ACTIONS
    tfidf_max_features: int = TFIDF_MAX_FEATURES
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = TFIDF_NGRAM_MAX
    min_cat_freq_ratio: float = MIN_NAV_CAT_FREQ_RATIO
    add_nav_interactions: bool = ADD_NAV_INTERACTIONS
    svd_n_components: int = SVD_N_COMPONENTS

@dataclass
class ArtifactsSummary:
    version: str
    config: dict
    nav_vocab: List[str]
    cat_maps: Dict[str, Dict[str, int]]
    label_map_present: bool
    tfidf_vectorizer_path: str
    svd_path: str
    top_actions: List[str]
    outlier_nb_actions_threshold: int

# ----------------------- Assembleur -----------------------
class FeaturesAssembler:
    def __init__(self, config: FeatureConfig):
        self.cfg = config
        self._pat_ecran = re.compile(r"\((.*?)\)")
        self._pat_conf = re.compile(r"<(.*?)>")
        self._pat_chaine = re.compile(r"\$(.*?)\$")
        self._tfidf: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._nav_vocab: List[str] = []
        self._cat_maps: Dict[str, Dict[str, int]] = {}
        self._label_map: Optional[Dict[str, int]] = None
        self._top_actions: List[str] = []
        self._outlier_thr: int = 0
        self._first_freq_map: Dict[str, float] = {}
        self._second_freq_map: Dict[str, float] = {}
        self._bigram_id_map: Dict[str, int] = {}

    # -------- FIT ----------
    def fit(self, df: pd.DataFrame, artifacts_dir: str):
        os.makedirs(artifacts_dir, exist_ok=True)

        # Label map
        if "util" in df.columns:
            cat = pd.Categorical(df["util"].fillna("").astype(str))
            self._label_map = {cat.categories[i]: int(i) for i in range(len(cat.categories))}
            pd.DataFrame({"util": cat.categories, "Y": range(len(cat.categories))}).to_csv(
                os.path.join(artifacts_dir, "label_mapping.csv"), index=False, encoding="utf-8"
            )
        else:
            self._label_map = None

        # Vocab navigateur
        if "navigateur" in df.columns:
            counts = df["navigateur"].fillna("").astype(str).value_counts(dropna=False)
            thr = max(1, int(self.cfg.min_cat_freq_ratio * len(df)))
            keep = counts[counts >= thr].index.tolist()
            if len(keep) < len(counts):
                keep.append("other")
            self._nav_vocab = keep
            with open(os.path.join(artifacts_dir, "nav_vocab.json"), "w", encoding="utf-8") as f:
                json.dump(self._nav_vocab, f, ensure_ascii=False)
        else:
            self._nav_vocab = []

        # Cat maps (entités extraites)
        ecran_raw  = df["trace_str"].apply(lambda x: plus_frequent(self._pat_ecran, x))
        conf_raw   = df["trace_str"].apply(lambda x: plus_frequent(self._pat_conf, x))
        chaine_raw = df["trace_str"].apply(lambda x: plus_frequent(self._pat_chaine, x))
        def learn_cmap(series: pd.Series) -> Dict[str, int]:
            categories = pd.Categorical(series.fillna("").astype(str)).categories.tolist()
            return {c: i for i, c in enumerate(categories)}
        self._cat_maps = {
            "ecran":  learn_cmap(ecran_raw),
            "conf":   learn_cmap(conf_raw),
            "chaine": learn_cmap(chaine_raw),
        }
        with open(os.path.join(artifacts_dir, "cat_maps.json"), "w", encoding="utf-8") as f:
            json.dump(self._cat_maps, f, ensure_ascii=False)

        # Top-K actions globales + documents (listes d'actions)
        compteur_global = Counter()
        action_docs = []
        for trace in df["trace_str"].astype(str):
            acts = actions_nettoyees_sans_t(trace)
            action_docs.append(acts)
            for a in acts:
                compteur_global[a] += 1
        self._top_actions = [a for a, _ in Counter(compteur_global).most_common(self.cfg.top_k_actions)]
        with open(os.path.join(artifacts_dir, "top_actions.json"), "w", encoding="utf-8") as f:
            json.dump(self._top_actions, f, ensure_ascii=False)

        # ===== TF-IDF (analyzer callable niveau module) =====
        self._tfidf = TfidfVectorizer(
            analyzer=ngram_actions,   # <— picklable
            preprocessor=None,
            tokenizer=None,
            lowercase=False,
            min_df=3,
            max_df=0.90,
            max_features=self.cfg.tfidf_max_features,
            sublinear_tf=True
        )
        self._tfidf.fit(action_docs)
        tfidf_path = os.path.join(artifacts_dir, "tfidf_vectorizer.joblib")
        joblib.dump(self._tfidf, tfidf_path)

        # ===== SVD topics =====
        tfidf_fit = self._tfidf.transform(action_docs)
        self._svd = TruncatedSVD(n_components=self.cfg.svd_n_components, random_state=42)
        self._svd.fit(tfidf_fit)
        svd_path = os.path.join(artifacts_dir, f"svd_{self.cfg.svd_n_components}.joblib")
        joblib.dump(self._svd, svd_path)

        # Fréquences 1ère/2ème action
        first_actions  = [acts[0] for acts in action_docs if len(acts) > 0]
        second_actions = [acts[1] for acts in action_docs if len(acts) > 1]
        def freq_map(lst: List[str]) -> Dict[str, float]:
            cnt = Counter(lst); tot = sum(cnt.values())
            return {k: v / tot for k, v in cnt.items()} if tot > 0 else {}
        self._first_freq_map  = freq_map(first_actions)
        self._second_freq_map = freq_map(second_actions)
        with open(os.path.join(artifacts_dir, "first_action_freq.json"), "w", encoding="utf-8") as f:
            json.dump(self._first_freq_map, f, ensure_ascii=False)
        with open(os.path.join(artifacts_dir, "second_action_freq.json"), "w", encoding="utf-8") as f:
            json.dump(self._second_freq_map, f, ensure_ascii=False)

        # Bigrammes d'actions compressés (ID)
        bigram_cnt = Counter()
        for acts in action_docs:
            bigram_cnt.update(bigrams_from_actions(acts))
        top_bigrams = [bg for bg, _ in bigram_cnt.most_common(TOP_K_ACTION_BIGRAMS)]
        self._bigram_id_map = {bg: i + 1 for i, bg in enumerate(top_bigrams)}  # 0 = autre/absent
        with open(os.path.join(artifacts_dir, "top_action_bigrams.json"), "w", encoding="utf-8") as f:
            json.dump(self._bigram_id_map, f, ensure_ascii=False)

        # Seuil outlier
        nb_actions_arr = np.where(df["trace_str"].astype(str) == "", 0, df["trace_str"].astype(str).str.count(",") + 1)
        self._outlier_thr = int(pd.Series(nb_actions_arr).quantile(0.99)) if len(df) > 0 else 0
        with open(os.path.join(artifacts_dir, "outlier_thr.json"), "w", encoding="utf-8") as f:
            json.dump({"nb_actions_q99": self._outlier_thr}, f, ensure_ascii=False)

        # Résumé artefacts
        summary = ArtifactsSummary(
            version="1.0.0",
            config=asdict(self.cfg),
            nav_vocab=self._nav_vocab,
            cat_maps=self._cat_maps,
            label_map_present=self._label_map is not None,
            tfidf_vectorizer_path=tfidf_path,
            svd_path=svd_path,
            top_actions=self._top_actions,
            outlier_nb_actions_threshold=self._outlier_thr
        )
        with open(os.path.join(artifacts_dir, "artifacts_summary.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    # --- LOAD ---
    def load(self, artifacts_dir: str):
        self._tfidf = joblib.load(os.path.join(artifacts_dir, "tfidf_vectorizer.joblib"))
        svd_path = os.path.join(artifacts_dir, f"svd_{self.cfg.svd_n_components}.joblib")
        self._svd = joblib.load(svd_path) if os.path.exists(svd_path) else None

        with open(os.path.join(artifacts_dir, "nav_vocab.json"), "r", encoding="utf-8") as f:
            self._nav_vocab = json.load(f)
        with open(os.path.join(artifacts_dir, "cat_maps.json"), "r", encoding="utf-8") as f:
            self._cat_maps = json.load(f)
        with open(os.path.join(artifacts_dir, "top_actions.json"), "r", encoding="utf-8") as f:
            self._top_actions = json.load(f)

        first_path  = os.path.join(artifacts_dir, "first_action_freq.json")
        second_path = os.path.join(artifacts_dir, "second_action_freq.json")
        bigram_path = os.path.join(artifacts_dir, "top_action_bigrams.json")
        if os.path.exists(first_path):
            with open(first_path, "r", encoding="utf-8") as f:
                self._first_freq_map = json.load(f)
        if os.path.exists(second_path):
            with open(second_path, "r", encoding="utf-8") as f:
                self._second_freq_map = json.load(f)
        if os.path.exists(bigram_path):
            with open(bigram_path, "r", encoding="utf-8") as f:
                self._bigram_id_map = json.load(f)

        thr_path = os.path.join(artifacts_dir, "outlier_thr.json")
        if os.path.exists(thr_path):
            with open(thr_path, "r", encoding="utf-8") as f:
                self._outlier_thr = json.load(f).get("nb_actions_q99", 0)

        label_map_path = os.path.join(artifacts_dir, "label_mapping.csv")
        if os.path.exists(label_map_path):
            df_map = pd.read_csv(label_map_path, dtype=str)
            self._label_map = {row["util"]: int(row["Y"]) for _, row in df_map.iterrows()}
        else:
            self._label_map = None

    # -------- TRANSFORM ----------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        trace = df["trace_str"].astype(str).fillna("")
        nb_actions = np.where(trace == "", 0, trace.str.count(",") + 1).astype(int)
        nb_time_tokens = trace.str.count(r"\bt\d+\b").astype(int)

        # t_max / durée
        extra_t = trace.str.extractall(r"\bt(\d+)\b")
        if not extra_t.empty:
            tmax_par_ligne = extra_t[0].astype(int).groupby(level=0).max()
            t_max = df.index.to_series().map(tmax_par_ligne).fillna(0).astype(int).values
        else:
            t_max = np.zeros(len(df), dtype=int)
        session_duration_sec = (self.cfg.seconds_per_tick * t_max).astype(int)

        # Actions & counts
        actions_lists = trace.apply(actions_nettoyees_sans_t)
        counts_lists = actions_lists.apply(Counter)
        n_act_clean = actions_lists.apply(len).astype(int)

        # Rythme
        dur = session_duration_sec.astype(float)
        rate = np.where(dur <= 0, 0.0, 60.0 * n_act_clean.values / np.clip(dur, 1e-6, None))

        # Entropie actions
        action_entropy = counts_lists.apply(entropy_from_counts).astype(float).values
        denom = np.log(np.clip(n_act_clean.values.astype(float), 1.0, None) + 1e-6)
        action_entropy_norm = np.where(denom > 0, action_entropy / denom, 0.0)

        # Diversité & concentration
        def _top_ratio(cnt: Counter, n_total: int):
            return float(max(cnt.values()) / n_total) if n_total > 0 and cnt else 0.0
        top_action_ratio = np.array([_top_ratio(c, n) for c, n in zip(counts_lists, n_act_clean)], dtype=float)
        top2_ratio = np.array([top2_ratio_from_counts(c, n) for c, n in zip(counts_lists, n_act_clean)], dtype=float)
        n_actions_uniques = counts_lists.apply(len).astype(int).values
        gini = np.array([gini_from_counts(c) for c in counts_lists], dtype=float)
        simpson = np.array([simpson_from_counts(c) for c in counts_lists], dtype=float)

        # Coverage TF (masqué)
        top_set = set(self._top_actions or [])
        vocab_covered = actions_lists.apply(lambda xs: sum(1 for x in xs if x in top_set)).astype(float).values
        n_total = n_act_clean.values.astype(float)
        coverage_rate = np.zeros_like(n_total, dtype=float)
        mask_cov = n_total > 0
        coverage_rate[mask_cov] = vocab_covered[mask_cov] / n_total[mask_cov]
        unknown_rate = 1.0 - coverage_rate

        # Ratios sémantiques
        texts_joined = actions_lists.apply(lambda toks: " ".join(toks)).astype(str)
        nb_ecrans = texts_joined.str.count("ecran").values
        nb_boutons = texts_joined.str.count("bouton").values
        nb_dialogues = texts_joined.str.count("dialogue").values
        nb_saisie = texts_joined.str.count("saisie").values
        nb_navigation = (
            texts_joined.str.count("chainage") +
            texts_joined.str.count("onglet") +
            texts_joined.str.count("retour sur un ecran") +
            texts_joined.str.count("navigation")
        ).values
        nb_raccourcis = texts_joined.str.count("raccourci").values

        def safe_ratio(x, total):
            total = np.asarray(total, dtype=float)
            out = np.zeros_like(total, dtype=float)
            mask = total > 0
            out[mask] = np.asarray(x, dtype=float)[mask] / total[mask]
            return out

        ratio_ecrans = safe_ratio(nb_ecrans, n_act_clean.values)
        ratio_boutons = safe_ratio(nb_boutons, n_act_clean.values)
        ratio_dialogues = safe_ratio(nb_dialogues, n_act_clean.values)
        ratio_saisie = safe_ratio(nb_saisie, n_act_clean.values)
        ratio_navigation = safe_ratio(nb_navigation, n_act_clean.values)
        ratio_raccourcis = safe_ratio(nb_raccourcis, n_act_clean.values)

        # Bins de durée
        session_length_bin = np.where(t_max <= 6, 0,
                                np.where(t_max <= 24, 1,
                                np.where(t_max <= 60, 2, 3))).astype(int)

        # Répétitions consécutives
        nb_repeat_sequences = actions_lists.apply(count_repeats).astype(int).values
        repeat_ratio = safe_ratio(nb_repeat_sequences, n_act_clean.values)

        # Retour arrière
        def back_count(xs: List[str]) -> int:
            if not xs: return 0
            c = 0
            for a in xs:
                a_low = a.lower()
                if ("retour" in a_low) or ("back" in a_low) or ("arriere" in a_low) or ("précédent" in a_low) or ("precedent" in a_low):
                    c += 1
            return c
        nb_back_actions = actions_lists.apply(back_count).astype(int).values
        back_action_ratio = safe_ratio(nb_back_actions, n_act_clean.values)

        # Position 1ère/2ème action
        first_actions = actions_lists.apply(lambda xs: xs[0] if len(xs) > 0 else "").values
        second_actions = actions_lists.apply(lambda xs: xs[1] if len(xs) > 1 else "").values
        first_action_freq = np.array([self._first_freq_map.get(str(a), 0.0) for a in first_actions], dtype=float)
        second_action_freq = np.array([self._second_freq_map.get(str(a), 0.0) for a in second_actions], dtype=float)

        # Bigramme compressé (ID top)
        def top_bigram_id(xs: List[str]) -> int:
            if len(xs) < 2 or not self._bigram_id_map:
                return 0
            bgs = bigrams_from_actions(xs)
            cnt = Counter(bgs)
            for bg, _ in cnt.most_common():
                if bg in self._bigram_id_map:
                    return int(self._bigram_id_map[bg])
            return 0
        top_bigram_ids = actions_lists.apply(top_bigram_id).astype(int).values

        # Fluidité temporelle
        def mean_var_actions_per_tick(ts: str) -> Tuple[float, float]:
            cps = actions_per_tick(ts)
            if len(cps) == 0:
                return 0.0, 0.0
            arr = np.array(cps, dtype=float)
            return float(arr.mean()), float(arr.var())
        mv = trace.apply(mean_var_actions_per_tick)
        avg_actions_per_tick = np.array([m for (m, v) in mv], dtype=float)
        var_actions_per_tick = np.array([v for (m, v) in mv], dtype=float)

        # std & burstiness
        def mean_std_actions_per_tick(ts: str) -> Tuple[float, float]:
            cps = actions_per_tick(ts)
            if len(cps) == 0:
                return 0.0, 0.0
            arr = np.array(cps, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0))
        ms = trace.apply(mean_std_actions_per_tick)
        mean_actions_per_tick = np.array([m for (m, s) in ms], dtype=float)
        std_actions_per_tick  = np.array([s for (m, s) in ms], dtype=float)
        burstiness_actions = np.where(
            (mean_actions_per_tick + std_actions_per_tick) > 0,
            (std_actions_per_tick - mean_actions_per_tick) / (std_actions_per_tick + mean_actions_per_tick + 1e-6),
            0.0
        )

        # Entités fréquentes (plus fréquent)
        ecran_raw  = trace.apply(lambda x: plus_frequent(self._pat_ecran, x)).astype(object)
        conf_raw   = trace.apply(lambda x: plus_frequent(self._pat_conf, x)).astype(object)
        chaine_raw = trace.apply(lambda x: plus_frequent(self._pat_chaine, x)).astype(object)
        def map_cat(raw: pd.Series, cmap: Dict[str, int]):
            return raw.apply(lambda x: cmap.get("" if x is None else str(x), -1)).astype(int).values
        ecran_plus_frequent  = map_cat(ecran_raw,  self._cat_maps["ecran"])
        conf_plus_frequent   = map_cat(conf_raw,   self._cat_maps["conf"])
        chaine_plus_frequent = map_cat(chaine_raw, self._cat_maps["chaine"])

        # Nav
        if "navigateur" in df.columns and self._nav_vocab:
            nav_raw = df["navigateur"].fillna("").astype(str).values
            known = set(self._nav_vocab)
            has_other = "other" in known
            nav_mapped = np.array([v if v in known else ("other" if has_other else v) for v in nav_raw], dtype=object)
            nav_cols = [f"nav_{c}" for c in self._nav_vocab]
            nav_matrix = np.zeros((len(df), len(self._nav_vocab)), dtype=int)
            col_idx = {c: i for i, c in enumerate(self._nav_vocab)}
            for i, v in enumerate(nav_mapped):
                j = col_idx.get(v, None)
                if j is not None:
                    nav_matrix[i, j] = 1
            nav_df = pd.DataFrame(nav_matrix, columns=nav_cols, index=df.index)
        else:
            nav_df = pd.DataFrame(index=df.index)

        if self.cfg.add_nav_interactions and not nav_df.empty:
            for c in list(nav_df.columns):
                nav_df[f"{c}__x_rate"] = nav_df[c].values * rate

        # ===== TF-IDF + Topics =====
        action_docs = actions_lists.tolist()         # liste de LISTES
        tfidf_mat = self._tfidf.transform(action_docs)
        tfidf_cols = [f"tfidf_{t}" for t in self._tfidf.get_feature_names_out()]
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_mat, index=df.index, columns=tfidf_cols)

        if self._svd is not None:
            tfidf_topics = self._svd.transform(tfidf_mat)
            topic_df = pd.DataFrame(
                {f"tfidf_topic_{i+1}": tfidf_topics[:, i] for i in range(tfidf_topics.shape[1])},
                index=df.index
            )
        else:
            topic_df = pd.DataFrame(index=df.index)

        is_outlier = (nb_actions > (self._outlier_thr or 0)).astype(int)

        base_df = pd.DataFrame({
            "nb_actions": nb_actions,
            "nb_time_tokens": nb_time_tokens,
            "t_max": t_max,
            "action_rate_per_min": rate,
            "action_entropy": action_entropy,
            "action_entropy_norm": action_entropy_norm,
            "top_action_ratio": top_action_ratio,
            "top2_ratio": top2_ratio,
            "ratio_ecrans": ratio_ecrans,
            "ratio_boutons": ratio_boutons,
            "ratio_dialogues": ratio_dialogues,
            "ratio_saisie": ratio_saisie,
            "ratio_navigation": ratio_navigation,
            "ratio_raccourcis": ratio_raccourcis,
            "n_actions_uniques": n_actions_uniques,
            "gini_diversity": gini,
            "simpson_concentration": simpson,
            "coverage_rate": coverage_rate,
            "unknown_rate": unknown_rate,
            "ecran_plus_frequent": ecran_plus_frequent,
            "conf_plus_frequent": conf_plus_frequent,
            "chaine_plus_frequent": chaine_plus_frequent,
            "is_outlier_nb_actions_q99": is_outlier,
            "session_length_bin": session_length_bin,
            "nb_repeat_sequences": nb_repeat_sequences,
            "repeat_ratio": repeat_ratio,
            "nb_back_actions": nb_back_actions,
            "back_action_ratio": back_action_ratio,
            "first_action_freq": first_action_freq,
            "second_action_freq": second_action_freq,
            "top_bigram_id": top_bigram_ids,
            "avg_actions_per_tick": avg_actions_per_tick,
            "var_actions_per_tick": var_actions_per_tick,
            "mean_actions_per_tick": mean_actions_per_tick,
            "std_actions_per_tick": std_actions_per_tick,
            "burstiness_actions": burstiness_actions,
        }, index=df.index)

        if self._label_map is not None and "util" in df.columns:
            Y = df["util"].fillna("").astype(str).apply(lambda u: self._label_map.get(u, -1)).astype(int)
            base_df.insert(0, "Y", Y.values)

        X = pd.concat([base_df, nav_df, tfidf_df, topic_df], axis=1)
        return X

# ----------------------- Principale fonction -----------------------
def feacture():
    print("🚀 Démarrage génération de features (mode VS Code)")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")
    if "trace_str" not in df.columns:
        raise ValueError("Colonne 'trace_str' manquante dans le CSV.")

    print(f"📊 Dimensions avant transformation : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    cfg = FeatureConfig()
    assembler = FeaturesAssembler(cfg)

    if TRANSFORM_ONLY:
        print("⚙️ Mode transform-only : chargement des artefacts…")
        assembler.load(ARTIFACTS_DIR)
    else:
        print("📌 Fit artefacts…")
        assembler.fit(df, ARTIFACTS_DIR)
        print("🔄 Rechargement artefacts…")
        assembler.load(ARTIFACTS_DIR)

    print("🛠 Transformation…")
    X = assembler.transform(df)

    print(f"✅ Dimensions après transformation : {X.shape[0]} lignes × {X.shape[1]} colonnes")

    X.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print("💾 Fichier de sortie enregistré :", OUTPUT_FILE)
    print("📁 Artefacts :", ARTIFACTS_DIR)

if __name__ == "__main__":
    feacture()

