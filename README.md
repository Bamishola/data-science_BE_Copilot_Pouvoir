# README — Data Science BE (Projet)

## Contexte & problématique
C'est dans le cadre du BE du MOD *Data Science* que s'inscrit ce projet.
Le contexte, les enjeux, la problématique et la démarche sont expliqué dans le rapport foruni.

## Prérequis
- Python 3.8+ installé sur la machine.
- accès terminal (Linux/macOS) ou PowerShell (Windows).
- [optionnel] Git si clonage depuis un dépôt distant.


## Installation & préparation de l'environnement virtuel
### Sous Linux/macOS
1. Se placer dans la racine du projet avec ```cd``` :

2. Créer l'environnement virtuel :
   ```
   python3 -m venv .venv
   ```

3. Activer l'environnement :
   ```
   source .venv/bin/activate
   ```

4. Mettre pip à jour :
   ```
   pip install --upgrade pip
   ```

5. Installer les dépendances dans le fichier ```requirements.txt``` :
   ```
   pip install -r requirements.txt
   ```

## Sous Windows PowerShell
- Création : ```python -m venv .venv```
- Activation : ```.\.venv\Scripts\Activate.ps1```
- Puis installer les dépendances spécifiées dans le fichier ```requirements.txt``` 

## Structure recommandée du projet
![Répertoire du projet](structure_projet.png)

## Exécution du fichier main (explication détaillée)

1. Pré-requis
   - L'environnement virtuel doit être activé.
   - Les fichiers d'entrée attendus `train.csv` et `test.csv` doivent se situer dans le répertoire `data`
   - Installer les dépendances : `pip install -r requirements.txt`.

2. Lancer le pipeline
   ```
   python main.py
   ```

3. Ce que fait `main.py` (ordre d'exécution)
   - Appelle `clean()` depuis `src.data_cleaning`
     - Nettoie les données (ex. `train.csv`) et génère `train_clean.csv`.
     - Message affiché : "---------Nettoyage des données en cours..." puis confirmation de génération.
   - Pause de 5 secondes (`time.sleep(5)`) pour laisser le fichier se stabiliser.
   - Appelle `feacture()` depuis `src.data_features` (nom de fonction tel qu'il est dans le code)
     - Génère des caractéristiques et crée `train_features_ready.csv`.
     - Message affiché : "---------Génération des features en cours..." puis confirmation.
   - Pause de 5 secondes.
   - Appelle `train_model()` depuis `src.train_model`
     - Entraîne le modèle (les artefacts produits dépendent de l'implémentation — vérifier `src/train_model.py` pour connaître le nom/chemin du modèle sauvegardé).
     - Message affiché : "---------Entraînement du modèle en cours..." puis confirmation.
   - Appelle `prediction_competition()` depuis `src.z_prediction_competiton`
     - Produit la prédiction finale pour la compétition (génère typiquement `submission.csv`).
     - Message affiché : "---------Début de la prédiction pour la compétition Kaggle..." puis confirmation de génération de `submission.csv`.

4. Fichiers attendus après exécution
   - train_clean.csv
   - train_features_ready.csv
   - artefacts d'entraînement (modèle, logs) — vérifier `src/train_model.py`
   - submission.csv (résultat des prédictions)

5. Conseils de dépannage
   - Si une étape échoue, consulter la sortie console pour le message d'erreur et ouvrir le fichier source correspondant dans `src/` pour plus de détails.
   - Vérifier que `train.csv`/`test.csv` existent et ont les colonnes attendues.
   - Si un module attend un chemin différent, adapter les chemins relatifs dans les scripts ou exécuter depuis la racine du projet.
   - Pour relancer proprement, supprimer les fichiers générés (ex. `train_clean.csv`, `train_features_ready.csv`, `submission.csv`) avant une nouvelle exécution.

Remarque : les noms de fonctions et de modules utilisés ici reflètent exactement le code fourni (p. ex. `feacture`, `z_prediction_competiton`). Corriger l'orthographe des noms si vous modifiez les fichiers sources.