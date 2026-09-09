# PROJET NLP 2 — ASSURVISION

## Analyse d'avis clients d'assurance

**Marcel Yammine & Thomas Wartelle | 2025–2026**

---

## Contenu du dossier

```text
Projet2/
├── Projet_NLP2_FINAL_Yammine_Wartelle.ipynb    # Notebook complet
├── app_insuranalytics.py                       # Application Streamlit
├── df_final.csv                                # Dataset final
├── df_clean.csv                                # Dataset nettoyé
├── df_complet.csv                              # Dataset avec traduction/résumé
├── tfidf_vectorizer.pkl                        # Modèle TF-IDF
├── lr_sentiment_model.pkl                      # Modèle de sentiment
├── lr_topic_model.pkl                          # Modèle de classification des topics
├── lr_stars_model.pkl                          # Modèle de prédiction des étoiles
├── resumes_assureur.csv                        # Résumés par assureur
├── Presentation_NLP2_AssurVision.pptx          # Présentation
├── video_presentation.mp4                      # Vidéo de présentation
├── avis_1_traduit.xlsx ... avis_35_traduit.xlsx # Données brutes
└── README.md                                   # Documentation du projet
```

---

## Important — Visualisation des graphiques

Les graphiques interactifs **Plotly** ne s'affichent pas nécessairement dans Jupyter Notebook en local.

Pour visualiser correctement l'ensemble des graphiques, il est recommandé d'ouvrir le notebook dans **Google Colab** :

1. Aller sur https://colab.research.google.com
2. Sélectionner `Fichier > Importer un notebook`
3. Uploader `Projet_NLP2_FINAL_Yammine_Wartelle.ipynb`

Les visualisations **Plotly** et **pyLDAvis** sont rendues dynamiquement et sont donc plus facilement accessibles depuis Google Colab ou un navigateur compatible.

---

# Option 1 — Lancer l'application en local

L'application fonctionne directement avec les fichiers fournis.  
Il n'est pas nécessaire de réexécuter le notebook.

### Étape 1 — Installer Python

Télécharger Python depuis :

https://www.python.org/downloads/

Lors de l'installation, cocher :

```text
Add Python to PATH
```

### Étape 2 — Installer les dépendances

Ouvrir un terminal (`cmd`) et exécuter :

```bash
pip install streamlit pandas numpy plotly matplotlib scikit-learn sentence-transformers
```

### Étape 3 — Vérifier les fichiers

Les fichiers suivants doivent être placés dans le même dossier :

```text
app_insuranalytics.py
df_final.csv
tfidf_vectorizer.pkl
lr_sentiment_model.pkl
lr_topic_model.pkl
lr_stars_model.pkl
resumes_assureur.csv
```

### Étape 4 — Lancer l'application

Dans un terminal ouvert dans le dossier du projet :

```bash
python -m streamlit run app_insuranalytics.py
```

L'application s'ouvrira dans le navigateur à l'adresse :

```text
http://localhost:8501
```

---

# Option 2 — Exécuter le notebook sur Google Colab

Le notebook fourni a déjà été exécuté et les résultats sont visibles.

Pour réexécuter certaines parties du pipeline :

### Étape 1 — Configurer Google Colab

Ouvrir le notebook dans Google Colab puis sélectionner :

```text
Runtime > Change runtime type > GPU T4
```

### Étape 2 — Uploader les datasets intermédiaires

Selon la partie à exécuter, uploader :

```text
df_clean.csv
```

pour reprendre à partir de la partie C, ou :

```text
df_complet.csv
```

pour reprendre à partir de la partie D ou E.

### Étape 3 — Reprendre à partir de la partie D : Embeddings

Ajouter cette cellule au début :

```python
import pandas as pd

df = pd.read_csv("df_complet.csv")
print(f"{df.shape[0]} lignes chargées")
```

Puis exécuter les cellules à partir de la **partie D**.

### Étape 4 — Reprendre à partir de la partie E : Supervised Learning

Charger de la même manière :

```python
import pandas as pd

df = pd.read_csv("df_complet.csv")
print(f"{df.shape[0]} lignes chargées")
```

Puis exécuter les cellules à partir de la **partie E**.

**Temps estimé : environ 40 minutes avec un GPU T4.**

> **Note :** les parties A (cleaning), B (traduction) et C (topic modeling) prennent environ 1 heure. Les fichiers CSV intermédiaires fournis permettent de sauter ces étapes.

---

# Option 3 — Lancer l'application Streamlit sur Google Colab

L'application AssurVision peut également être lancée directement depuis Google Colab grâce à **ngrok**.

### Étape 1 — Uploader les fichiers

Uploader dans le panneau de fichiers de Google Colab :

```text
app_insuranalytics.py
df_final.csv
tfidf_vectorizer.pkl
lr_sentiment_model.pkl
lr_topic_model.pkl
lr_stars_model.pkl
resumes_assureur.csv
```

### Étape 2 — Créer un compte ngrok

Créer un compte gratuit sur :

https://ngrok.com

ngrok permet de créer un tunnel donnant temporairement accès à l'application Streamlit exécutée sur Google Colab.

### Étape 3 — Récupérer le token ngrok

Accéder à :

https://dashboard.ngrok.com/authtokens

Puis copier le token affiché.

### Étape 4 — Ajouter le token

Dans la dernière cellule du notebook, remplacer :

```python
NGROK_TOKEN = "VOTRE_TOKEN_ICI"
```

par votre token personnel :

```python
NGROK_TOKEN = "2abc123xyz..."
```

### Étape 5 — Lancer l'application

Exécuter la dernière cellule du notebook.

Un lien similaire à celui-ci sera généré :

```text
https://xxxx.ngrok-free.dev
```

Cliquer sur ce lien pour accéder à l'application AssurVision.

---

# Application AssurVision

L'application comporte **6 onglets principaux**.

## 1. Prédiction

Saisir un avis client afin d'obtenir :

- le sentiment prédit : **positif ou négatif** ;
- le nombre d'étoiles prédit : **1 à 5** ;
- le thème détecté.

## 2. Résumé

Sélectionner un assureur pour afficher :

- sa note moyenne ;
- son nombre d'avis ;
- son pourcentage d'avis positifs ;
- un résumé généré par IA ;
- différentes visualisations.

## 3. Explication

Saisir un avis afin de visualiser les **15 mots les plus influents** dans la prédiction du modèle :

- vert : influence positive ;
- rouge : influence négative.

## 4. Recherche

Explorer les avis grâce à différents filtres :

- assureur ;
- note ;
- thème ;
- mot-clé.

## 5. RAG — Retrieval-Augmented Generation

Poser une question en langage naturel.

Le système :

1. recherche les avis les plus pertinents ;
2. récupère le contexte correspondant ;
3. génère une synthèse à partir des informations retrouvées.

## 6. QA — Question Answering

Poser des questions telles que :

```text
Quel est le meilleur assureur ?
```

Le système fournit une réponse structurée accompagnée de graphiques lorsque cela est pertinent.

---

# Vidéo de présentation

Le fichier :

```text
video_presentation.mp4
```

contient une présentation du projet couvrant :

- le pipeline NLP complet, des parties A à F ;
- une démonstration de l'application AssurVision ;
- les principaux résultats ;
- leur interprétation.

---

# Résultats principaux

| Élément | Résultat |
|---|---:|
| Dataset | **34 415 avis** |
| Nombre d'assureurs | **50+** |
| Note moyenne | **2,85 / 5** |
| Dataset rééquilibré | **9 731 positifs + 9 731 négatifs** |
| Meilleur modèle de sentiment | **Random Forest + TF-IDF** |
| Accuracy sentiment | **84,8 %** |
| Prédiction des étoiles | **53,4 %** |
| MAE étoiles | **0,61** |
| Modèles comparés | **9** |

Les approches étudiées incluent notamment :

**TF-IDF, Embeddings, CNN, LSTM, USE et CamemBERT.**

---

## Auteurs

**Marcel Yammine**  
**Thomas Wartelle**

Projet NLP 2 — **2025–2026**
