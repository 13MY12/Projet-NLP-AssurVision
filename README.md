PROJET NLP 2 — ASSURVISION
Analyse d'avis clients d'assurance
Marcel Yammine & Thomas Wartelle | 2025-2026



CONTENU DU DOSSIER


Projet2/
  Projet_NLP2_FINAL_Yammine_Wartelle.ipynb   -> Notebook complet
  app_insuranalytics.py                       -> Application Streamlit
  df_final.csv                                -> Dataset final
  df_clean.csv                                -> Dataset nettoye
  df_complet.csv                              -> Dataset avec traduction/resume
  tfidf_vectorizer.pkl                        -> Modele TF-IDF
  lr_sentiment_model.pkl                      -> Modele sentiment
  lr_topic_model.pkl                          -> Modele topics
  lr_stars_model.pkl                          -> Modele etoiles
  resumes_assureur.csv                        -> Resumes par assureur
  Presentation_NLP2_AssurVision.pptx          -> Presentation
  video_presentation.mp4                      -> Video de presentation (7 min)
  avis_1_traduit.xlsx ... avis_35_traduit.xlsx -> Donnees brutes
  README.txt                                  -> Ce fichier


================================================================
IMPORTANT : VISUALISATION DES GRAPHIQUES
================================================================

Les graphiques interactifs (Plotly) ne s'affichent PAS dans
Jupyter Notebook en local. Pour voir tous les graphiques :

-> Ouvrir le notebook dans Google Colab :
   https://colab.research.google.com
   Fichier > Importer un notebook > Upload

Les graphiques Plotly et pyLDAvis sont rendus dynamiquement
et ne sont visibles que dans Google Colab ou dans un
navigateur compatible.


================================================================
OPTION 1 : LANCER L'APPLICATION EN LOCAL (LE PLUS SIMPLE)
================================================================

L'application fonctionne directement avec les fichiers fournis.
Pas besoin de re-executer le notebook.

Etape 1 : Installer Python si pas deja fait
   https://www.python.org/downloads/
   Cocher "Add Python to PATH" pendant l'installation

Etape 2 : Installer les dependances
   Ouvrir un terminal (cmd) et taper :
   pip install streamlit pandas numpy plotly matplotlib scikit-learn sentence-transformers

Etape 3 : Mettre ces fichiers dans le meme dossier :
   - app_insuranalytics.py
   - df_final.csv
   - tfidf_vectorizer.pkl
   - lr_sentiment_model.pkl
   - lr_topic_model.pkl
   - lr_stars_model.pkl
   - resumes_assureur.csv

Etape 4 : Lancer l'application
   Ouvrir un terminal dans le dossier et taper :
   python -m streamlit run app_insuranalytics.py

   L'application s'ouvre dans le navigateur a http://localhost:8501


================================================================
OPTION 2 : EXECUTER LE NOTEBOOK SUR GOOGLE COLAB
================================================================

Le notebook a deja ete execute et les resultats sont visibles.
Si vous souhaitez re-executer certaines parties :

Etape 1 : Ouvrir le notebook dans Google Colab
   Runtime > Change runtime type > GPU T4

Etape 2 : Uploader les fichiers CSV fournis
   - df_clean.csv (pour demarrer a partir de la partie C)
   - df_complet.csv (pour demarrer a partir de la partie D ou E)

Etape 3 : Pour re-executer a partir de la partie D (Embeddings) :
   Ajouter cette cellule au debut :

   import pandas as pd
   df = pd.read_csv('df_complet.csv')
   print(f"{df.shape[0]} lignes chargees")

   Puis executer les cellules a partir de la partie D.

Etape 4 : Pour re-executer a partir de la partie E (Supervised Learning) :
   Meme chose, charger df_complet.csv puis executer a partir de la partie E.
   Temps estime : 40 minutes avec GPU T4.

Note : Les parties A (cleaning), B (traduction) et C (topic modeling)
prennent environ 1h. Les fichiers CSV fournis permettent de les sauter.


================================================================
OPTION 3 : LANCER L'APPLICATION STREAMLIT SUR GOOGLE COLAB
================================================================

Si vous souhaitez lancer l'application AssurVision depuis Colab :

Etape 1 : Uploader ces fichiers dans Colab (panneau fichiers a gauche) :
   - app_insuranalytics.py
   - df_final.csv
   - tfidf_vectorizer.pkl
   - lr_sentiment_model.pkl
   - lr_topic_model.pkl
   - lr_stars_model.pkl
   - resumes_assureur.csv

Etape 2 : Creer un compte gratuit sur https://ngrok.com
   (ngrok est un service de tunneling qui cree un lien public
   temporaire pour acceder a l'application sur Colab)

Etape 3 : Recuperer votre token ngrok :
   - Aller sur https://dashboard.ngrok.com/authtokens
   - Copier le token affiche

Etape 4 : Dans la derniere cellule du notebook, remplacer
   NGROK_TOKEN = "VOTRE_TOKEN_ICI"
   par votre token personnel, exemple :
   NGROK_TOKEN = "2abc123xyz..."

Etape 5 : Executer la derniere cellule du notebook.
   Un lien s'affiche (ex: https://xxxx.ngrok-free.dev)
   Cliquer dessus pour ouvrir l'application.



================================================================
DESCRIPTION DE L'APPLICATION ASSURVISION (6 onglets)
================================================================

1. PREDICTION
   Saisir un avis client -> obtenir le sentiment (positif/negatif),
   les etoiles predites (1-5) et le theme detecte.

2. RESUME
   Selectionner un assureur -> voir les metriques (note moyenne,
   nombre d'avis, % positifs), le resume IA et des graphiques.

3. EXPLICATION
   Saisir un avis -> voir les 15 mots les plus influents
   (vert = positif, rouge = negatif).

4. RECHERCHE
   Filtrer les avis par assureur, note, theme et mot-cle.

5. RAG (Retrieval-Augmented Generation)
   Poser une question en langage naturel -> le systeme retrouve
   les avis pertinents et genere une synthese automatique.

6. QA (Question-Answering)
   Poser des questions comme "quel est le meilleur assureur ?"
   et obtenir une reponse structuree avec graphiques.


================================================================
VIDEO DE PRESENTATION
================================================================

Le fichier video_presentation.mp4 contient une presentation
de 5 minutes couvrant :
- Le pipeline NLP complet (parties A a F)
- La demonstration live de l'application AssurVision
- Les resultats et l'interpretation


================================================================
RESULTATS PRINCIPAUX
================================================================

Meilleur modele sentiment : Random Forest + TF-IDF -> 84.8%
Prediction etoiles (5 classes) : 53.4% (MAE = 0.61)
Dataset : 34 415 avis, 50+ assureurs, note moyenne 2.85/5
Dataset reequilibre : 9 731 positifs + 9 731 negatifs
9 modeles compares (TF-IDF, Embedding, CNN, LSTM, USE, CamemBERT)
