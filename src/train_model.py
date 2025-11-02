import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib  
from pathlib import Path


PROJECT_ROOT = Path.cwd()
INPUT = PROJECT_ROOT / "data" / "train_features_ready.csv" # chemin du fichier d'entrée des features
MODEL_FILE = PROJECT_ROOT / "data" / "rf_model.joblib"  # nom du fichier pour le modèle sauvegardé

TEST_SIZE = 0.20
RANDOM_STATE = 42

RF_KW = dict(
    n_estimators=800,
    criterion="gini",
    max_depth=None,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=2,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
)

def ensure_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0.0)

def train_model():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"Fichier introuvable: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)
    print(f"📥 Chargé: {INPUT}")
    print(f"🔹 Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    if "Y" not in df.columns:
        raise ValueError("La colonne 'Y' (labels) est absente du CSV.")

    y = df["Y"].astype(int).values
    X = df.drop(columns=["Y"])
    X = ensure_numeric(X)

    # ===== Split =====
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(**RF_KW)
    rf.fit(X_train, y_train)

    # ===== Évaluation =====
    y_pred = rf.predict(X_val)
    print("\n===== Résultats (RandomForest) =====")
    print(f"Accuracy    : {accuracy_score(y_val, y_pred):.4f}")
    print(f"F1-Macro    : {f1_score(y_val, y_pred, average='macro'):.4f}")
    print(f"F1-Weighted : {f1_score(y_val, y_pred, average='weighted'):.4f}")
    print(f"OOB Score   : {getattr(rf, 'oob_score_', np.nan):.4f}")
    print(classification_report(y_val, y_pred, digits=3, zero_division=0))
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_val, y_pred))

    # ===== Importance des features =====
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop 25 features :")
    print(imp_df.head(25).to_string(index=False))

    # ===== Sauvegarde du modèle =====
    joblib.dump(rf, MODEL_FILE)
    print(f"\n✅ Modèle sauvegardé dans : {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
