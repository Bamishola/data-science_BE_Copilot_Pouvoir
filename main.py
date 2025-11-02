from src.data_cleaning import clean
from src.data_features import feacture
from src.train_model import train_model
from src.z_prediction_competiton import prediction_competition
import time


if __name__ == "__main__":

    # Appel de la fonction clean dans le fichier data_cleaning.py
    # Nettoie train.csv et crée train_clean.csv avec colonnes 'util', 'navigateur' et 'trace_str'
    print("---------Nettoyage des données en cours...")
    clean()
    print("---------Nettoyage terminé. Fichier train_clean.csv généré.")

    # Attendre 5 secondes pour s'assurer que le fichier est généré
    print("--------- Pause de 5 seconde avant de générer les features...")
    time.sleep(5)

    # Appel de la fonction feacture dans le fichier data_features.py
    # Génère des caractéristiques supplémentaires et crée train_features_ready.csv
    print("---------Génération des features en cours...")
    feacture()
    print("---------Génération des features terminée. Fichier train_features_ready.csv généré.")

    # Attendre 5 secondes pour s'assurer que le fichier est généré
    print("--------- Pause de 5 seconde avant l'entrainement du modèle...")
    time.sleep(5)

    # Appel de la fonction train_model dans le fichier train_model.py
    print("---------Entraînement du modèle en cours...")
    train_model()
    print("---------Entraînement du modèle terminé.")




    # Prediction pour competition Kaggle submission.csv avec test.csv
    print("---------Début de la prédiction pour la compétition Kaggle...")
    prediction_competition()
    print("---------Prédiction terminée. Fichier submission.csv généré.")