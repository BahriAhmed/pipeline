import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

"""
Fonction pour executer le nettoyage des données jusqu'à
la sauvegarde du modèle de ML optimisé.
"""
def pipeline():
    # Charger les données
    filepath = os.path.join("..", "data", "Airline_Dataset.csv")
    data = pd.read_csv(filepath)

    # Étape 1 : Renommer les colonnes
    data.rename(columns=lambda x: x.strip().replace(" ", "_").lower(), inplace=True)

    # Étape 2 : Gérer les valeurs manquantes
    data["arrival_delay_in_minutes"] = data["arrival_delay_in_minutes"].fillna(
        data["arrival_delay_in_minutes"].median()
    )

    # Étape 3 : Combiner les retards
    data["total_delay"] = data["departure_delay_in_minutes"] + data["arrival_delay_in_minutes"]

    # Étape 4 : Étiqueter la satisfaction
    data["satisfaction"] = data["satisfaction"].apply(lambda x: 1 if x == "satisfied" else 0)

    # Étape 5 : Supprimer les colonnes non représentatives
    columns_to_drop = ["id", "departure_delay_in_minutes", "arrival_delay_in_minutes"]
    data.drop(columns=columns_to_drop, inplace=True)

    # Étape 6 : Identifier les colonnes catégoriques et numériques
    categorical_columns = data.select_dtypes(include=["object"]).columns
    numerical_columns = data.select_dtypes(include=["number"]).drop(columns=["satisfaction"]).columns

    # Étape 7 : Configurer le prétraitement
    categorical_preprocessor = OneHotEncoder(drop="first", sparse_output=False)
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_preprocessor, numerical_columns),
            ("cat", categorical_preprocessor, categorical_columns),
        ]
    )

    # Étape 8 : Diviser les données en caractéristiques (X) et cible (y)
    X = data.drop(columns=["satisfaction"])
    y = data["satisfaction"]

    # Étape 9 : Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Étape 10 : Définir le modèle choisi avec ses hyperparamètres optimisés
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            learning_rate=0.03400676560252863,
            max_iter=925,
            max_depth=13,
            min_samples_leaf=16,
            l2_regularization=8.763352346928274,
            max_bins=209,
            random_state=42
        ))
    ])

    # Étape 11 : Entraîner et évaluer le modèle
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Étape 12 : Sauvegarder le modèle avec joblib
    model_save_path = "optimized_model.pkl"
    joblib.dump(model_pipeline, model_save_path)
    print(f"Le modèle optimisé a été sauvegardé dans le fichier : {model_save_path}")

"""
Execute le pipeline.
"""
if __name__ == "__main__":
    try:
        pipeline()
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")