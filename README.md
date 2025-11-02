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
   - Appelle `clean()` depuis `src.data_cleaning` : Nettoie `train.csv` et génère `train_clean.csv` qui est un nouveau dataset contenant trois colonnes (utilisateurs, navigateurs, et actions)
   - Pause de 5 secondes pour s'assurer que `train_clean.csv` est bien généré.
   - Appelle `feacture()` depuis `src.data_features` : Génère de nouvelles caractéristiques, fait l' encodage (toutes les colonnes deviennent numériques) et crée `train_features_ready.csv`.
   - Pause de 5 secondes pour s'assurer que `train_features_ready.csv` est bien généré.
   - Appelle `train_model()` depuis `src.train_model` : Entraîne le modèle  et la sauvergarder pour pouvoir faire de la prédiction en temps réel plus tard.
   - Appelle `prediction_competition()` depuis `src.z_prediction_competiton`: Produit la prédiction finale pour la compétition (génère typiquement `submission.csv`). C' est ce fichier qui sera soumis sur Kaggle pour la compétition.
   

4. Fichiers principaux attendus après exécution dans le répertoire `data`
   - `train_clean.csv`
   - `train_features_ready.csv`
   - Modèle sauvergardé `rf_model.jolib`
   - `submission.csv` :  résultat des prédictions (pour Kaggle) à partir du fichier `test.csv`

5. Conseils de dépannage
   - Si une étape échoue, consulter la sortie console pour le message d'erreur.
   - Vérifier que `train.csv`/`test.csv` existent et ont les colonnes attendues.
   - Si un module attend un chemin différent, adapter les chemins relatifs dans les scripts ou exécuter depuis la racine du projet.
   - Pour relancer proprement, supprimer les fichiers générés avant une nouvelle exécution.


## Membre de l'équipe,
- LOKE Bamishola Aristide
- MAMOUDOU Wone
- EZZAKOUDY Taybi