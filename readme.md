# Optimized Machine Learning Pipeline for Airline Satisfaction

## Description
Ce projet implémente un pipeline complet pour :
- Nettoyer et prétraiter les données,
- Entraîner un modèle de machine learning optimisé,
- Évaluer ses performances,
- Sauvegarder le modèle dans un fichier pour une utilisation future.

Le modèle utilisé est un **HistGradientBoostingClassifier**, configuré avec des hyperparamètres optimisés.

---

## Prérequis

### Environnement Python
1. Assurez-vous que **Python (version >= 3.7)** est installé. Vérifiez votre version avec :
   ```bash
   python --version
   ```
2. Le fichier `Airline_Dataset.csv` doit être dans le dossier `data`
3. Se trouver dans le dossier `models` avant d'executer le fichier `pipeline.py`

---

## Configuration de l'environnement

### Création et activation
1. Créez un environnement virtuel :
   ```bash
   python -m venv .venv
   ```
2. Activez l'environnement :
   - Sur **Windows** :
     ```bash
     .venv\Scripts\activate
     ```
   - Sur **Mac/Linux** :
     ```bash
     source .venv/bin/activate
     ```

3. Mettez à jour `pip` :
   ```bash
   pip install --upgrade pip
   ```

---

## Installation des dépendances

Installez les bibliothèques nécessaires au projet :
```bash
pip install pandas scikit-learn joblib
```

---

## Exécution du script

### Étapes
1. Assurez-vous que `Airline_Dataset.csv` est dans le dossier `data` et que vous vous situez dans le dossier `models` au chemin suivant `../pipeline/models`
2. Activez votre environnement virtuel.
3. Lancez le script avec la commande suivante :
   ```bash
   python pipeline.py
   ```

### Résultats attendus
- Un rapport de classification (précision, rappel, f1-score) s’affichera dans la console.
- Un fichier `optimized_model.pkl` contenant le modèle entraîné sera généré.

---

## Structure du projet

Organisation des fichiers :
```
data
|  ├── Airline_Dataset.csv    # Dataset pour l'entraînement du modèle
|
models
|  ├── pipeline.py            # Script Python contenant le pipeline de ML
   ├── optimized_model.pkl    # Modèle entraîné sauvegardé
|
.gitignore                    # Contient le .venv à ignorer lors des "git push"
README.md                     # Documentation du projet
```

---

## Résolution des problèmes

1. **Dataset introuvable** :
   - Vérifiez que `Airline_Dataset.csv` est dans le bon dossier.
2. **No such file or directory** :
   - Vérifier que vous vous trouvez bien dans le dossier `models` au moment d'executer le fichier `pipeline.py`
3. **Problème d'installation des librairies** :
   - Mettez à jour la bibliothèque concernée avec :
     ```bash
     pip install package_name --upgrade
     ```
4. **Problème avec l'environnement virtuel** :
   - Vérifiez que l'environnement est activé avant d’exécuter les commandes.

---
