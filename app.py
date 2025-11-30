import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Prédiction du risque de CHD",
    page_icon="🫀",
    layout="centered"
)

st.title("🩺 Application de prédiction du risque de maladie cardiaque (CHD)")
st.write("""
Cette application web a été **développée avec VS Code** et déployée avec **Streamlit**.  
Elle utilise un modèle de Machine Learning déjà entraîné et sauvegardé dans `Model.pkl`
(pipeline : prétraitement + ACP + régression logistique) à partir du dataset **CHD.csv**.
""")
# 👉 Affichage de l’image du cœur
st.image("coeur.jpg", width=200)


# ---------------------------------------------------
# 1. Chargement du modèle
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("Model.pkl")  # même nom que dans train_model.py
    return model

model = load_model()


# ---------------------------------------------------
# 2. Formulaire de saisie des variables
# ---------------------------------------------------
st.subheader("🧾 Saisir les informations du patient")

with st.form("chd_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=10, max_value=100, value=50)
        sbp = st.number_input("Pression systolique (sbp)", min_value=80.0, max_value=250.0, value=140.0)
        ldl = st.number_input("LDL (mauvais cholestérol)", min_value=0.0, max_value=1000.0, value=200.0)

    with col2:
        adiposity = st.number_input("Adiposity", min_value=0.0, max_value=10000.0, value=2500.0)
        obesity = st.number_input("Obesity", min_value=0.0, max_value=6000.0, value=3000.0)
        famhist = st.selectbox("Antécédents familiaux (famhist)", ["Present", "Absent"])

    submitted = st.form_submit_button("Prédire le risque")


# ---------------------------------------------------
# 3. Prédiction avec le modèle
# ---------------------------------------------------
if submitted:
    # Construire un DataFrame avec les colonnes EXACTES du modèle
    input_data = {
        "sbp": sbp,
        "ldl": ldl,
        "adiposity": adiposity,
        "obesity": obesity,
        "age": age,
        "famhist": famhist
    }

    input_df = pd.DataFrame([input_data])

    st.write("### Données saisies")
    st.dataframe(input_df)

    # Prédictions
    proba_chd = model.predict_proba(input_df)[0, 1]
    pred_chd = model.predict(input_df)[0]

    st.subheader("🩺 Résultat de la prédiction")
    st.write(f"**Probabilité estimée de CHD (classe 1)** : `{proba_chd:.2f}`")

    if pred_chd == 1:
        st.error("🔴 Le modèle prédit **un risque ÉLEVÉ** de maladie cardiaque (CHD = 1).")
    else:
        st.success("🟢 Le modèle prédit **un risque FAIBLE** de maladie cardiaque (CHD = 0).")

    st.info("⚕️ Cette application est pédagogique et ne remplace pas un avis médical.")
