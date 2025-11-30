# ============================================================
# Entraînement complet du modèle CHD + sauvegarde Model.pkl
# Dataset : CHD.csv (séparateur ; )
# Pipeline : Imputer + StandardScaler + OneHotEncoder + PCA + LogisticRegression
# ============================================================

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ------------------------------------------------------------
# 1. Chargement du dataset
# ------------------------------------------------------------
df = pd.read_csv("CHD.csv", sep=";")

print("Colonnes :", df.columns)
print(df.head())

# ------------------------------------------------------------
# 2. Harmonisation de la colonne catégorielle famhist
# ------------------------------------------------------------
df["famhist"] = df["famhist"].astype(str).str.strip().str.lower()

df["famhist"] = df["famhist"].replace({
    "present": "Present",
    "absent": "Absent"
})

print("Catégories harmonisées :", df["famhist"].unique())

# ------------------------------------------------------------
# 3. Cible et variables explicatives
# ------------------------------------------------------------
y = df["chd"]
X = df.drop("chd", axis=1)

# ------------------------------------------------------------
# 4. Variables numériques et catégorielles
# ------------------------------------------------------------
numeric_features = ["sbp", "ldl", "adiposity", "obesity", "age"]
categorical_features = ["famhist"]

# ------------------------------------------------------------
# 5. Préprocesseurs avec SimpleImputer
# ------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ------------------------------------------------------------
# 6. PIPELINE COMPLET
# ------------------------------------------------------------
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=5)),
    ("logreg", LogisticRegression(max_iter=500))
])

# ------------------------------------------------------------
# 7. Train-test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 8. Entraînement
# ------------------------------------------------------------
pipeline.fit(X_train, y_train)

# ------------------------------------------------------------
# 9. Évaluation
# ------------------------------------------------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n===============================")
print("Accuracy sur le test :", acc)
print("===============================\n")

# ------------------------------------------------------------
# 10. Réentraînement sur tout le dataset
# ------------------------------------------------------------
pipeline.fit(X, y)

# ------------------------------------------------------------
# 11. Sauvegarde du modèle final
# ------------------------------------------------------------
joblib.dump(pipeline, "Model.pkl")

print("✔✔✔ Model.pkl sauvegardé avec succès !")
