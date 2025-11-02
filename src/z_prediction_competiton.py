# src/01_data_cleaning.py (ajouter en plus de clean_train_file)
import os
from pathlib import Path
import time
import pandas as pd
from src.data_cleaning import read_csv_any, normalize_headers, drop_useless_columns, find_col_exact, find_col_fuzzy, parse_single_column_lines, BAD_TOKENS, USER_COL_CANDIDATES, BROWSER_COL_CANDIDATES
from src.data_features import FeatureConfig, FeaturesAssembler
import joblib
import numpy as np
from src.train_model import ensure_numeric  # on réutilise ta fonction

# ----------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------
PROJECT_ROOT = Path.cwd()
TRANSFORM_ONLY = False

INPUT = PROJECT_ROOT / "data" / "test.csv" # chemin du fichier d'entrée des features
INPUT_FILE_2 = PROJECT_ROOT / "data" / "test_clean.csv" # chemin du fichier d'entrée des features


OUTPUT = PROJECT_ROOT / "data" / "test_clean.csv"  # nom du fichier pour le modèle sauvegardé
OUTPUT_FILE_2 = PROJECT_ROOT / "data" / "test_features_ready.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts_vscode"

MODEL_FILE = PROJECT_ROOT / "data" / "rf_model.joblib"
TEST_FEATURES_FILE = PROJECT_ROOT / "data" / "test_features_ready.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "submission.csv"
LABEL_ENCODER_OUTPUT = ARTIFACTS_DIR / "label_mapping.csv"

def clean_test_file(input_file: Path, output_file: Path, user_col_arg=None, browser_col_arg=None):
    """
    Nettoyage complet du CSV de test.
    Même logique que clean_train_file, mais pas de colonne Y.
    """
    df, used_sep = read_csv_any(input_file)
    print(f"Lecture test OK : {len(df)} lignes × {len(df.columns)} colonnes (sep='{used_sep}')")
    df = normalize_headers(df)

    # Cas 1 : une seule colonne
    if df.shape[1] == 1:
        sole_col = df.columns[0]
        out = parse_single_column_lines(df, sole_col)
        out.to_csv(output_file, index=False, encoding="utf-8")
        print(f"✅ Fichier test généré : {output_file}")
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
        raise RuntimeError("Aucune colonne d'actions détectée dans le fichier test.")

    trace_str = df[action_cols].apply(
        lambda row: ", ".join([v.strip() for c, v in row.items() 
                               if c in action_cols and isinstance(v, str) and v.strip().lower() not in BAD_TOKENS]),
        axis=1
    )

    out = pd.DataFrame({"trace_str": trace_str})
    out.insert(0, "navigateur", df[nav_col].astype(str).fillna("unknown") if nav_col else "unknown")
    if user_col:
        out.insert(0, "util", df[user_col].astype(str))

    out.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Fichier test généré : {output_file} ({len(out)} lignes × {len(out.columns)} colonnes)")

# Petite fonction pour simplifier l'appel
def clean_test(input_path=INPUT, output_path=OUTPUT):
    clean_test_file(Path(input_path), Path(output_path))


# fonction de génération des features
def feature():
    print("🚀 Démarrage génération de features (mode VS Code)")

    if not os.path.exists(INPUT_FILE_2):
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE_2}")

    df = pd.read_csv(INPUT_FILE_2, dtype=str).fillna("")

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

    # ================== ALIGNEMENT COLONNES AVEC LE MODELE ==================
    # Charger le modèle sauvegardé
    model = joblib.load(MODEL_FILE)

    # Colonnes exactes utilisées à l'entraînement
    train_cols = model.feature_names_in_

    # Supprimer colonnes test absentes dans train
    extra_cols = set(X.columns) - set(train_cols)
    if extra_cols:
        print(f"⚠️ Suppression des colonnes non présentes dans le train : {len(extra_cols)}")
        X.drop(columns=list(extra_cols), inplace=True)

    # Ajouter colonnes du train absentes dans le test
    missing_cols = set(train_cols) - set(X.columns)
    if missing_cols:
        print(f"ℹ️ Ajout des colonnes manquantes du train : {len(missing_cols)}")
        for col in missing_cols:
            X[col] = 0  # valeur neutre pour le ML

    # Réordonner les colonnes exactement comme dans le train
    X = X[train_cols]

    print(f"✅ Dimensions après alignement : {X.shape[0]} lignes × {X.shape[1]} colonnes")

    X.to_csv(OUTPUT_FILE_2, index=False, encoding="utf-8")
    print("💾 Fichier de sortie enregistré :", OUTPUT_FILE_2)
    print("📁 Artefacts :", ARTIFACTS_DIR)

# ----------------------------------------------------------
# FONCTION DE PRÉDICTION
# ----------------------------------------------------------
def predict_competition():
    # Vérification des fichiers
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_FILE}")
    if not os.path.exists(TEST_FEATURES_FILE):
        raise FileNotFoundError(f"Fichier test introuvable : {TEST_FEATURES_FILE}")

    # Chargement du modèle
    print(f"📦 Chargement du modèle depuis {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)

    # Chargement des features du test
    print(f"📥 Chargement des features test depuis {TEST_FEATURES_FILE}")
    df_test = pd.read_csv(TEST_FEATURES_FILE, low_memory=False)
    print(f"🔹 Données test : {df_test.shape[0]} lignes × {df_test.shape[1]} colonnes")

    # Vérifier si une colonne identifiant existe
    id_col = None
    for candidate in ["id", "ID", "util", "user", "user_id"]:
        if candidate in df_test.columns:
            id_col = candidate
            break

    # Conversion numérique
    X_test = df_test.copy()
    if id_col:
        X_test = X_test.drop(columns=[id_col])
    X_test = ensure_numeric(X_test)

    # Prédiction
    print("🤖 Prédiction en cours...")
    y_pred = model.predict(X_test)

    # --- Décodage des labels si label_mapping existe ---
    label_map_path = LABEL_ENCODER_OUTPUT
    if os.path.exists(label_map_path):
        df_map = pd.read_csv(label_map_path, dtype=str)
        # inverse du mapping : {0: 'util1', 1: 'util2', ...}
        inv_label_map = {int(row["Y"]): row["util"] for _, row in df_map.iterrows()}
        y_pred_labels = [inv_label_map.get(int(y), "unknown") for y in y_pred]
    else:
        y_pred_labels = y_pred  # reste en int si pas de mapping

    # Création du dataframe de soumission
    if id_col:
        submission = pd.DataFrame({id_col: df_test[id_col], "Y": y_pred_labels})
    else:
        submission = pd.DataFrame({"id": np.arange(len(y_pred_labels)), "Y": y_pred_labels})

    # Sauvegarde
    submission.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✅ Fichier de soumission généré : {OUTPUT_FILE}")
    print(submission.head())




def prediction_competition():

    print("--------------- Competition Kaggle Prediction -----------------")
    """
    Fonction principale pour nettoyer le fichier de test dans le cadre de la compétition de prédiction.
    """
    clean_test()

    # Pause pour s'assurer que le fichier est généré
    print("--------- Pause de 5 secondes avant de générer les features...")
    time.sleep(5)

    # Génération des features pour le fichier de test
    feature()

    # Pause pour s'assurer que le fichier est généré
    print("--------- Pause de 5 secondes avant la prédiction des labels...")
    time.sleep(5)


    # Prédiction des labels pour le fichier de test
    predict_competition()




if __name__ == "__main__":
    prediction_competition()