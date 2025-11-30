Projet : Application Streamlit de prédiction du risque de maladie cardiaque (CHD)

Description :
Cette application utilise un modèle de Machine Learning entraîné (Model.pkl) pour prédire le risque de maladie cardiaque à partir de variables cliniques. L’interface est développée avec Streamlit.

Fichiers inclus :
- app.py : script principal de l'application Streamlit
- Model.pkl : modèle ML entraîné (pipeline complet : imputation + normalisation + PCA + régression logistique)
- coeur.jpg : image affichée dans l'interface
- train_model.py : script d'entraînement du modèle à partir du fichier CHD.csv
- CHD.csv : (optionnel) dataset utilisé pour entraîner le modèle
- requirements.txt : liste des dépendances nécessaires

Instructions d'exécution (Windows) :

1. Créer un environnement virtuel :
   python -m venv venv

2. Activer l'environnement virtuel :
   venv\Scripts\activate

3. Installer les dépendances :
   pip install -r requirements.txt

4. Lancer l'application Streamlit :
   streamlit run app.py

Notes :
- Le modèle Model.pkl contient tout le pipeline prétraitement + PCA + régression logistique.
