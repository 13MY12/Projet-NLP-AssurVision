import streamlit as st
import pickle, os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AssurVision", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0EA5E9 0%, #10B981 100%);
        padding: 2.5rem 2rem; border-radius: 16px; color: white;
        text-align: center; margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.3);
    }
    .main-header h1 { color: white; font-size: 2.8rem; font-weight: 700; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.15rem; margin: 0.5rem 0 0 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #f0fdf4; padding: 6px; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 8px; padding: 10px 20px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #0EA5E9 !important; color: white !important; }
    .stButton > button { background: linear-gradient(135deg, #0EA5E9, #10B981); color: white; border: none; border-radius: 8px; padding: 0.5rem 2rem; font-weight: 600; }
    .footer { text-align: center; color: #94a3b8; padding: 2rem 0 1rem 0; font-size: 0.85rem; border-top: 1px solid #e2e8f0; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔍 AssurVision</h1>
    <p>Plateforme d'analyse intelligente des avis d'assurance</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f: tfidf = pickle.load(f)
    with open("lr_sentiment_model.pkl", "rb") as f: sent_model = pickle.load(f)
    with open("lr_topic_model.pkl", "rb") as f: topic_model = pickle.load(f)
    return tfidf, sent_model, topic_model

@st.cache_data
def load_data(): return pd.read_csv("df_final.csv")

@st.cache_data
def load_resumes():
    try: return pd.read_csv("resumes_assureur.csv")
    except: return pd.DataFrame()

try:
    tfidf, sent_model, topic_model = load_models()
    df = load_data()
    df_resumes = load_resumes()
    models_loaded = True
except Exception as e:
    st.error(f"Erreur : {e}")
    models_loaded = False

if models_loaded:
    import plotly.express as px

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Prediction", "📊 Resume", "🔍 Explication",
        "🔎 Recherche", "🤖 RAG", "❓ QA"
    ])

    # ━━━ TAB 1: PREDICTION ━━━
    with tab1:
        st.header("🔮 Prediction de sentiment, etoiles et theme")
        avis_input = st.text_area("Votre avis :", height=120,
            placeholder="Ex: Le remboursement a ete tres rapide et le service client excellent.", key="pred_input")

        if st.button("🔍 Analyser", key="btn_pred") and avis_input.strip():
            vec = tfidf.transform([avis_input])
            prob = sent_model.predict_proba(vec)[0]
            is_pos = prob[1] > 0.5
            confiance = max(prob)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Sentiment")
                if is_pos:
                    st.markdown("<div style='background:linear-gradient(135deg,#f0fdf4,#dcfce7);padding:1.5rem;border-radius:12px;text-align:center;border:2px solid #10B981;'><span style='font-size:3rem;'>😊</span><h2 style='color:#10B981;margin:0.5rem 0 0 0;'>Positif</h2></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background:linear-gradient(135deg,#fef2f2,#fee2e2);padding:1.5rem;border-radius:12px;text-align:center;border:2px solid #EF4444;'><span style='font-size:3rem;'>😞</span><h2 style='color:#EF4444;margin:0.5rem 0 0 0;'>Negatif</h2></div>", unsafe_allow_html=True)
                st.progress(float(confiance))
                st.caption(f"Confiance : {confiance:.1%}")

            with col2:
                st.subheader("Etoiles predites")
                try:
                    with open("lr_stars_model.pkl", "rb") as f:
                        stars_model = pickle.load(f)
                    stars_pred = stars_model.predict(vec)[0]
                    stars_proba = stars_model.predict_proba(vec)[0]
                    st.markdown(f"<div style='background:linear-gradient(135deg,#fffbeb,#fef3c7);padding:1.5rem;border-radius:12px;text-align:center;border:2px solid #f59e0b;'><span style='font-size:2.5rem;'>{'⭐' * int(stars_pred)}</span><h2 style='color:#f59e0b;margin:0.5rem 0 0 0;'>{int(stars_pred)} / 5</h2></div>", unsafe_allow_html=True)
                    proba_stars = pd.DataFrame({"Etoiles": [1,2,3,4,5], "Probabilite": stars_proba})
                    fig_s = px.bar(proba_stars, x="Etoiles", y="Probabilite", color="Probabilite", color_continuous_scale=["#fef3c7", "#f59e0b"])
                    fig_s.update_layout(height=250, showlegend=False, margin=dict(l=0,r=0,t=0,b=0), coloraxis_showscale=False)
                    st.plotly_chart(fig_s, use_container_width=True)
                except:
                    st.info("lr_stars_model.pkl non disponible")

            with col3:
                st.subheader("Theme detecte")
                try:
                    topic_pred = topic_model.predict(vec)[0]
                    topic_proba = topic_model.predict_proba(vec)[0]
                    st.markdown(f"<div style='background:linear-gradient(135deg,#ecfeff,#e0f2fe);padding:1rem;border-radius:12px;text-align:center;border:2px solid #0EA5E9;'><h3 style='color:#0EA5E9;margin:0;'>{topic_pred}</h3></div>", unsafe_allow_html=True)
                    proba_df = pd.DataFrame({"Theme": topic_model.classes_, "Probabilite": topic_proba})
                    proba_df = proba_df[proba_df["Probabilite"] > 0.03].sort_values("Probabilite", ascending=True)
                    fig = px.bar(proba_df, x="Probabilite", y="Theme", orientation="h", color="Probabilite", color_continuous_scale=["#e0f2fe", "#0EA5E9"])
                    fig.update_layout(height=250, showlegend=False, margin=dict(l=0,r=0,t=0,b=0), coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Modele thematique non disponible.")

    # ━━━ TAB 2: RESUME ━━━
    with tab2:
        st.header("📊 Resume par assureur")
        assureur_sel = st.selectbox("Assureur", sorted(df["assureur"].dropna().unique().tolist()), key="resume_a")
        dff = df[df["assureur"] == assureur_sel]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Note moyenne", f"{dff['note'].mean():.1f}/5")
        c2.metric("Nb avis", len(dff))
        pct = (dff.get('sentiment_label', pd.Series()) == 'Positif').mean()
        c3.metric("% Positifs", f"{pct:.0%}" if pct > 0 else "N/A")
        c4.metric("5 etoiles", (dff['note'] == 5).sum())

        if not df_resumes.empty:
            row = df_resumes[df_resumes['assureur'] == assureur_sel]
            if not row.empty:
                st.subheader("Resume IA")
                st.info(row.iloc[0]['resume_en'])

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig = px.histogram(dff, x='note', title=f"Notes — {assureur_sel}", color_discrete_sequence=['#0EA5E9'])
            st.plotly_chart(fig, use_container_width=True)
        with col_g2:
            if 'topic_lda' in dff.columns:
                tc = dff['topic_lda'].value_counts().head(7)
                if len(tc) > 0:
                    fig2 = px.pie(values=tc.values, names=tc.index, title="Themes", color_discrete_sequence=px.colors.sequential.Teal)
                    st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(dff[['note', 'avis_clean']].head(20), use_container_width=True)

    # ━━━ TAB 3: EXPLICATION ━━━
    with tab3:
        st.header("🔍 Explication des predictions")
        avis_expl = st.text_area("Avis :", height=120, key="expl_input")

        if st.button("🔍 Expliquer", key="btn_expl") and avis_expl.strip():
            vec = tfidf.transform([avis_expl])
            prob = sent_model.predict_proba(vec)[0]
            is_pos = prob[1] > 0.5
            st.markdown(f"**Prediction : {'😊 Positif' if is_pos else '😞 Negatif'}** — Confiance : {max(prob):.1%}")

            feature_names = tfidf.get_feature_names_out()
            coefs = sent_model.coef_[0]
            vec_array = vec.toarray()[0]
            present_indices = np.where(vec_array > 0)[0]

            if len(present_indices) > 0:
                contributions = coefs[present_indices] * vec_array[present_indices]
                words = [feature_names[i] for i in present_indices]
                df_c = pd.DataFrame({"Mot": words, "Contribution": contributions})
                df_c = df_c.reindex(df_c["Contribution"].abs().sort_values(ascending=False).index).head(15)
                df_c = df_c.sort_values("Contribution")

                colors = ["#10B981" if v > 0 else "#EF4444" for v in df_c["Contribution"]]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(df_c["Mot"], df_c["Contribution"], color=colors)
                ax.axvline(0, color="#64748b", lw=0.8)
                ax.set_title("Top 15 mots influents", fontsize=14, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(facecolor='#10B981', label='Positif'), Patch(facecolor='#EF4444', label='Negatif')], loc='lower right')
                plt.tight_layout()
                st.pyplot(fig)

                top_pos = df_c[df_c["Contribution"] > 0].tail(3)["Mot"].tolist()
                top_neg = df_c[df_c["Contribution"] < 0].head(3)["Mot"].tolist()
                if top_pos: st.success(f"**Mots positifs** : {', '.join(top_pos)}")
                if top_neg: st.error(f"**Mots negatifs** : {', '.join(top_neg)}")


    # ━━━ TAB 4: RECHERCHE ━━━
    with tab4:
        st.header("🔎 Recherche d'information")
        c1, c2, c3 = st.columns(3)
        with c1: a_f = st.selectbox("Assureur", ["Tous"] + sorted(df["assureur"].dropna().unique().tolist()), key="r_a")
        with c2: n_r = st.slider("Notes", 1, 5, (1, 5), key="r_n")
        with c3:
            tl = ["Tous"] + (sorted([x for x in df["topic_lda"].dropna().unique().tolist() if x != '']) if "topic_lda" in df.columns else [])
            t_f = st.selectbox("Theme", tl, key="r_t")

        kw = st.text_input("Mot-cle", placeholder="remboursement, sinistre...", key="r_kw")

        dff = df.copy()
        if a_f != "Tous": dff = dff[dff["assureur"] == a_f]
        dff = dff[dff["note"].between(*n_r)]
        if t_f != "Tous": dff = dff[dff["topic_lda"] == t_f]
        if kw: dff = dff[dff["avis_clean"].str.contains(kw, case=False, na=False)]

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Resultats", len(dff))
        c2.metric("Note moy.", f"{dff['note'].mean():.2f}" if len(dff) > 0 else "N/A")
        c3.metric("Assureurs", dff["assureur"].nunique())

        cols = [c for c in ["assureur", "note", "topic_lda", "sentiment_label", "avis_clean"] if c in dff.columns]
        st.dataframe(dff[cols].head(100), use_container_width=True, height=400)

        if len(dff) > 0:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig = px.histogram(dff, x='note', title="Distribution des notes (filtre)", color_discrete_sequence=['#0EA5E9'])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col_g2:
                if 'sentiment_label' in dff.columns:
                    sent_counts = dff['sentiment_label'].value_counts()
                    fig2 = px.pie(values=sent_counts.values, names=sent_counts.index, title="Sentiment",
                                  color_discrete_map={'Positif': '#10B981', 'Negatif': '#EF4444'})
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, use_container_width=True)

    # ━━━ TAB 5: RAG (COMPLET) ━━━
    with tab5:
        st.header("🤖 RAG — Retrieval-Augmented Generation")
        st.markdown("Posez une question en langage naturel. Le systeme **retrouve** les avis pertinents puis **genere** une reponse synthetique.")

        question = st.text_input("Question :", placeholder="Que pensent les clients du remboursement chez AXA ?", key="rag_q")
        c1, c2 = st.columns(2)
        with c1: n_res = st.slider("Nombre d'avis a retrouver", 3, 20, 10, key="rag_n")
        with c2: r_a = st.selectbox("Filtrer par assureur", ['Tous'] + sorted(df['assureur'].dropna().unique().tolist()), key="rag_a")

        if st.button("🔍 Rechercher et generer", key="btn_rag") and question.strip():
            with st.spinner("Recherche semantique et generation de reponse..."):
                try:
                    from sentence_transformers import SentenceTransformer
                    from sklearn.metrics.pairwise import cosine_similarity

                    @st.cache_resource
                    def load_emb(): return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                    emb = load_emb()

                    dr = df.dropna(subset=['avis_clean'])
                    if r_a != 'Tous': dr = dr[dr['assureur'] == r_a]
                    dr = dr.sample(min(5000, len(dr)), random_state=42)

                    c_emb = emb.encode(dr['avis_clean'].tolist(), show_progress_bar=False)
                    q_emb = emb.encode([question])
                    scores = cosine_similarity(q_emb, c_emb)[0]
                    top_idx = scores.argsort()[-n_res:][::-1]
                    res = dr.iloc[top_idx].copy()
                    res['score'] = scores[top_idx]

                    st.markdown("---")

                    # ━━━ GENERATION : Synthese automatique ━━━
                    st.subheader("📝 Reponse generee")

                    note_moy = res['note'].dropna().mean()
                    nb_pos = (res.get('sentiment_label', pd.Series()) == 'Positif').sum()
                    nb_neg = (res.get('sentiment_label', pd.Series()) == 'Negatif').sum()
                    
                    # Si pas de sentiment_label, utiliser la note pour determiner le sentiment
                    if nb_pos + nb_neg == 0:
                        nb_pos = (res['note'] >= 4).sum()
                        nb_neg = (res['note'] <= 2).sum()
                        
                    nb_total = len(res)
                    best_assureurs = res.groupby('assureur')['note'].mean().sort_values(ascending=False)

                    # Extraire les mots-cles frequents dans les resultats
                    all_words = ' '.join(res['avis_clean'].tolist()).lower().split()
                    stop_words = set(['les', 'des', 'une', 'pour', 'que', 'qui', 'est', 'pas', 'dans', 'plus', 'par', 'avec', 'sur', 'mon', 'mais', 'très', 'bien', 'fait', 'tout', 'sont'])
                    word_freq = pd.Series([w for w in all_words if len(w) > 3 and w not in stop_words]).value_counts().head(10)

                    # Generer la reponse textuelle
                    if note_moy >= 3.5:
                        sentiment_txt = "majoritairement positifs"
                    elif note_moy <= 2.5:
                        sentiment_txt = "majoritairement negatifs"
                    else:
                        sentiment_txt = "partages"
                    filtre_txt = f" pour {r_a}" if r_a != 'Tous' else ""

                    reponse = f"""**Analyse basee sur {nb_total} avis pertinents{filtre_txt} :**

Les avis sont **{sentiment_txt}** avec une note moyenne de **{note_moy:.1f}/5**.
- {nb_pos} avis positifs et {nb_neg} avis negatifs sur {nb_total} avis retrouves.
"""
                    if len(best_assureurs) > 1:
                        top3 = best_assureurs.head(3)
                        reponse += f"\n**Assureurs les mieux notes sur ce sujet :**\n"
                        for assureur, note in top3.items():
                            reponse += f"- {assureur} : {note:.1f}/5\n"

                    reponse += f"\n**Mots-cles les plus frequents dans les avis :** {', '.join(word_freq.index[:7])}"

                    st.info(reponse)

                    # ━━━ RETRIEVAL : Avis sources ━━━
                    st.subheader(f"📋 Avis sources ({nb_total} retrouves)")

                    for _, row in res.iterrows():
                        badge = "🟢" if row.get('sentiment_label') == 'Positif' else "🔴"
                        note_val = row.get('note', 0)
                        stars = "⭐" * int(note_val) if pd.notna(note_val) else ""
                        with st.expander(f"{badge} {row.get('assureur','')} | {stars} | Pertinence: {row['score']:.1%}"):
                            st.write(row['avis_clean'])
                            if 'topic_lda' in row and pd.notna(row.get('topic_lda')) and row.get('topic_lda') != '':
                                st.caption(f"Theme : {row['topic_lda']}")

                    # Metriques
                    st.subheader("📊 Metriques")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Note moyenne", f"{note_moy:.1f}/5" if pd.notna(note_moy) else "N/A")
                    c2.metric("Sentiment dominant", "Positif" if nb_pos > nb_neg else "Negatif")
                    c3.metric("Pertinence max", f"{res['score'].max():.1%}")

                except Exception as e: st.error(f"Erreur : {e}")

    # ━━━ TAB 6: QA (AMELIORE) ━━━
    with tab6:
        st.header("❓ Question-Answering")
        st.markdown("Posez une question et obtenez une reponse basee sur l'analyse des avis clients.")

        # Suggestions avec selectbox au lieu de boutons
        suggestion = st.selectbox("💡 Questions suggerees :", [
            "(tapez votre propre question)",
            "Quel est le meilleur assureur ?",
            "Quel assureur eviter ?",
            "Statistiques globales",
            "Avis sur le remboursement",
            "Problemes de resiliation",
            "Service client telephone"
        ], key="qa_suggestion")

        default_q = suggestion if suggestion != "(tapez votre propre question)" else ""
        q = st.text_input("Question :", value=default_q, placeholder="Quel assureur recommander ?", key="qa_q")

        if st.button("🔍 Repondre", key="btn_qa") and q.strip():
            ql = q.lower()
            st.markdown("---")

            if any(w in ql for w in ['meilleur', 'top', 'recommand', 'bien']):
                st.subheader("🏆 Meilleurs assureurs")
                best = df.groupby('assureur').agg(
                    note_moy=('note', 'mean'),
                    nb_avis=('note', 'count')
                ).sort_values('note_moy', ascending=False)
                best = best[best['nb_avis'] >= 10].head(10)

                # Reponse textuelle
                top1 = best.index[0]
                st.info(f"**Le meilleur assureur est {top1}** avec une note moyenne de {best.iloc[0]['note_moy']:.1f}/5 "
                        f"sur {int(best.iloc[0]['nb_avis'])} avis. "
                        f"Les 3 meilleurs sont : {', '.join(best.index[:3])}.")

                fig = px.bar(best.reset_index(), x='note_moy', y='assureur', orientation='h',
                             color='note_moy', color_continuous_scale=['#fee2e2', '#10B981'], range_color=[1,5],
                             title="Top 10 assureurs par note moyenne")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            elif any(w in ql for w in ['pire', 'eviter', 'mauvais', 'fuir']):
                st.subheader("⚠️ Assureurs a eviter")
                worst = df.groupby('assureur').agg(
                    note_moy=('note', 'mean'),
                    nb_avis=('note', 'count')
                ).sort_values('note_moy')
                worst = worst[worst['nb_avis'] >= 10].head(10)

                bottom1 = worst.index[0]
                st.warning(f"**L'assureur le moins bien note est {bottom1}** avec une note moyenne de {worst.iloc[0]['note_moy']:.1f}/5 "
                           f"sur {int(worst.iloc[0]['nb_avis'])} avis.")

                fig = px.bar(worst.reset_index(), x='note_moy', y='assureur', orientation='h',
                             color='note_moy', color_continuous_scale=['#EF4444', '#fecaca'], range_color=[1,5],
                             title="10 assureurs les moins bien notes")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            elif any(w in ql for w in ['combien', 'statistique', 'total', 'globale']):
                st.subheader("📊 Statistiques globales")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total avis", f"{len(df):,}")
                c2.metric("Assureurs", df['assureur'].nunique())
                c3.metric("Note moyenne", f"{df['note'].mean():.2f}")
                if 'sentiment_label' in df.columns:
                    c4.metric("% Positifs", f"{(df['sentiment_label']=='Positif').mean():.0%}")

                fig = px.histogram(df, x='note', title="Distribution globale des notes", color_discrete_sequence=['#0EA5E9'])
                st.plotly_chart(fig, use_container_width=True)

                st.info(f"**Resume** : Le dataset contient {len(df):,} avis sur {df['assureur'].nunique()} assureurs. "
                        f"La note moyenne est de {df['note'].mean():.1f}/5. "
                        f"Les notes 1 et 5 sont les plus frequentes, ce qui montre une polarisation des opinions.")

            else:
                # Recherche par mots-cles dans les avis
                words = [w for w in ql.split() if len(w) > 3]
                if words:
                    pattern = '|'.join(words)
                    dq = df[df['avis_clean'].str.contains(pattern, case=False, na=False)]
                    if len(dq) > 0:
                        note_moy = dq['note'].mean()
                        nb_pos = (dq.get('sentiment_label', pd.Series()) == 'Positif').sum()
                        nb_neg = len(dq) - nb_pos

                        # Reponse textuelle
                        sentiment_txt = "positifs" if nb_pos > nb_neg else "negatifs"
                        st.info(f"**{len(dq)} avis trouves** sur le sujet \"{q}\".\n\n"
                                f"Note moyenne : **{note_moy:.1f}/5** | "
                                f"Sentiment dominant : **{sentiment_txt}** ({nb_pos} positifs, {nb_neg} negatifs).\n\n"
                                f"Les assureurs les plus mentionnes : {', '.join(dq['assureur'].value_counts().head(3).index.tolist())}.")

                        # Meilleurs assureurs sur ce sujet
                        top_assureurs = dq.groupby('assureur')['note'].mean().sort_values(ascending=False).head(5)
                        st.subheader("Meilleurs assureurs sur ce sujet")
                        for assureur, note in top_assureurs.items():
                            st.write(f"- **{assureur}** : {note:.1f}/5 {'⭐' * round(note)}")

                        st.subheader("Exemples d'avis")
                        st.dataframe(dq[['assureur', 'note', 'avis_clean']].head(10), use_container_width=True)
                    else:
                        st.warning("Aucun resultat. Essayez avec d'autres mots-cles.")
                else:
                    st.warning("Question trop courte. Ajoutez plus de details.")

    st.markdown('<div class="footer"><p>🔍 <strong>AssurVision</strong> — Analyse intelligente des avis d\'assurance | Projet NLP 2 | 2026</p></div>', unsafe_allow_html=True)