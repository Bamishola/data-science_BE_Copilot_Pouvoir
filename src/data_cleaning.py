# src/01_data_cleaning.py
"""
Nettoyage initial du fichier train.csv et création de train_clean.csv.

Colonnes finales :
- 'util' : identifiant utilisateur (si présent)
- 'navigateur' : nom du navigateur
- 'trace_str' : chaîne concaténant toutes les actions de l'utilisateur, séparées par des virgules
"""

from pathlib import Path
import pandas as pd
import os
import zipfile
import subprocess
import argparse


# -------------------------------------------------------
# Constantes
# -------------------------------------------------------
BAD_TOKENS = {"", "nan", "none", "null"}

USER_COL_CANDIDATES = [
    "util", "user", "utilisateur", "label",
    "USER", "USER_ID", "user_id", "id_user", "idutil"
]

BROWSER_COL_CANDIDATES = [
    "navigateur", "browser", "nav", "user_agent", "agent_navigateur"
]

BROWSER_KEYWORDS = [
    "chrome", "firefox", "edge", "internet explorer",
    "ie", "safari", "opera", "brave", "vivaldi", "yandex"
]

# -------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------

def read_csv_any(path: Path):
    """Tente de lire un CSV avec différents séparateurs et encodages."""
    last_err = None
    for sep in [",", ";", "\t"]:
        for enc in [None, "utf-8-sig"]:
            try:
                df = pd.read_csv(path, sep=sep, dtype=str, encoding=enc, low_memory=False, header=None)
                return df, sep
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Impossible de lire {path}. Dernière erreur : {last_err}")

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les caractères invisibles et espaces dans les noms de colonnes."""
    df.columns = [("" if c is None else str(c).replace("\ufeff", "").strip()) for c in df.columns]
    return df

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime colonnes vides, dupliquées ou avec des valeurs invalides."""
    before = df.shape[1]
    drop_cols = []
    for c in df.columns:
        s = df[c]
        if not s.notna().any():
            drop_cols.append(c)
            continue
        vals = s.dropna().astype(str).str.strip().str.lower()
        if (vals.isin(BAD_TOKENS)).all():
            drop_cols.append(c)
    drop_cols += [c for c in df.columns if c.lower().startswith("unnamed:")]
    df = df.drop(columns=list(set(drop_cols)), errors="ignore")
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    print(f"🧹 Colonnes supprimées : {before - df.shape[1]}")
    return df

def find_col_exact(candidates, cols):
    """Retourne la première colonne exacte trouvée parmi les candidats."""
    for c in candidates:
        if c in cols:
            return c
    return None

def find_col_fuzzy(cols, keywords):
    """Retourne la première colonne correspondant approximativement aux mots-clés."""
    norm = {c: c.strip().lower() for c in cols}
    for c, lc in norm.items():
        if any(k in lc for k in keywords):
            return c
    return None

def is_browser_token(tok: str) -> bool:
    t = tok.strip().lower()
    if not t:
        return False
    return any(k in t for k in BROWSER_KEYWORDS)

def build_trace_from_tokens(tokens):
    """Concatène les tokens valides en une chaîne séparée par des virgules."""
    toks = [t.strip() for t in tokens if t is not None and str(t).strip() != ""]
    return ", ".join(toks)

def parse_single_column_lines(df_onecol: pd.DataFrame, colname: str):
    """Parse une seule colonne contenant chaînes tokenisées par des virgules."""
    lines = df_onecol[colname].astype(str).tolist()
    util_list, nav_list, trace_list = [], [], []
    for line in lines:
        tokens = [t.strip() for t in line.split(",")]
        if not tokens:
            util_list.append(None); nav_list.append("unknown"); trace_list.append("")
            continue
        if is_browser_token(tokens[0]):
            util = None
            nav = tokens[0]
            trace_tokens = tokens[1:]
        else:
            util = tokens[0] if tokens[0] else None
            if len(tokens) >= 2 and is_browser_token(tokens[1]):
                nav = tokens[1]
                trace_tokens = tokens[2:]
            else:
                nav = "unknown"
                trace_tokens = tokens[1:]
        util_list.append(util)
        nav_list.append(nav if nav else "unknown")
        trace_list.append(build_trace_from_tokens(trace_tokens))
    out = pd.DataFrame({"navigateur": nav_list, "trace_str": trace_list})
    if any(u is not None and str(u).strip() != "" for u in util_list):
        out.insert(0, "util", [u if u is not None else "" for u in util_list])
    return out

# -------------------------------------------------------
# Fonctions principales
# -------------------------------------------------------
def clean_train_file(input_file: Path, output_file: Path, user_col_arg=None, browser_col_arg=None):
    """Pipeline complet de nettoyage du CSV d'entraînement."""
    df, used_sep = read_csv_any(input_file)
    print(f"Lecture OK : {len(df)} lignes × {len(df.columns)} colonnes (sep='{used_sep}')")
    df = normalize_headers(df)

    # Cas 1 : une seule colonne
    if df.shape[1] == 1:
        sole_col = df.columns[0]
        out = parse_single_column_lines(df, sole_col)
        out.to_csv(output_file, index=False, encoding="utf-8")
        print(f"✅ Fichier généré : {output_file}")
        return

    # Cas 2 : plusieurs colonnes
    df = drop_useless_columns(df)
    cols = df.columns.tolist()

    user_col = user_col_arg or find_col_exact(USER_COL_CANDIDATES, cols) or find_col_fuzzy(cols, USER_COL_CANDIDATES)
    nav_col = browser_col_arg or find_col_exact(BROWSER_COL_CANDIDATES, cols) or find_col_fuzzy(cols, BROWSER_COL_CANDIDATES)

    print(f"Utilisateur : {user_col if user_col else '(absent)'}")
    print(f"Navigateur : {nav_col if nav_col else '(unknown)'}")

    exclude = set(filter(None, [user_col, nav_col]))
    action_cols = [c for c in cols if c not in exclude]
    if not action_cols:
        raise RuntimeError("Aucune colonne d'actions détectée.")

    # Construction trace_str
    trace_str = df[action_cols].apply(
        lambda row: ", ".join([v.strip() for c, v in row.items() if c in action_cols and isinstance(v, str) and v.strip().lower() not in BAD_TOKENS]),
        axis=1
    )

    out = pd.DataFrame({"trace_str": trace_str})
    out.insert(0, "navigateur", df[nav_col].astype(str).fillna("unknown") if nav_col else "unknown")
    if user_col:
        out.insert(0, "util", df[user_col].astype(str))

    out.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Fichier généré : {output_file} ({len(out)} lignes × {len(out.columns)} colonnes)")

# -------------------------------------------------------
# Exécution en script
# -------------------------------------------------------
def clean():
    parser = argparse.ArgumentParser(description="Nettoyage CSV train.csv → train_clean.csv")
    parser.add_argument("--input", default="data/train.csv")
    parser.add_argument("--output", default="data/train_clean.csv")
    args = parser.parse_args()
    in_file = Path(args.input).resolve()
    out_file = Path(args.output).resolve()
    clean_train_file(in_file, out_file)

if __name__ == "__main__":
    clean()