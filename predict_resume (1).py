# Excelr_resume_dashboard.py
"""
ExcelR Resume Dashboard
- Multi-upload resumes
- Dark/Light theme fully applied
- Aggregated & per-resume EDA
- Animated skill trends & networks
- SHAP explanations included (best-effort)
- Interactive Plotly visuals & WordClouds
"""

import os
import re
import math
import time
import joblib
import string
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from datetime import datetime
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# SHAP support (optional)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ----------------------------- Page Config -----------------------------
st.set_page_config(
    page_title="🚀 Excelr Resume Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling -----------------------------
BASE_FONT = "Poppins, sans-serif"
st.markdown(f"""
<style>
html, body {{ font-family: {BASE_FONT}; }}
h1,h2,h3,h4 {{ font-weight:700; }}
.section-card {{
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}}
.small-muted {{ color: #9aa; font-size:12px; }}
.badge {{ display:inline-block; padding:6px 12px; margin:4px; border-radius:12px; color:white; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Sidebar -----------------------------
st.sidebar.title("⚙️ Settings")
theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
max_resumes = st.sidebar.slider("Max resumes to upload", min_value=1, max_value=25, value=8)
show_shap = st.sidebar.checkbox("Show SHAP explanations (may be slow)", value=False)
shap_samples = st.sidebar.slider("SHAP: background samples", min_value=10, max_value=200, value=50, step=10)

CHART_TEMPLATE = "plotly_dark" if theme_choice == "Dark" else "plotly_white"
WC_BG = "#0f1724" if theme_choice == "Dark" else "white"
TEXT_COLOR = "#EEE" if theme_choice == "Dark" else "#111"

# ----------------------------- IdentityPreprocessor (fallback) -----------------------------
class IdentityPreprocessor:
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    def fit_transform(self, X, y=None): return X

# ----------------------------- Load Artifacts (cached) -----------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts["voting_clf"] = joblib.load("models/voting_ensemble_trained.pkl")
    artifacts["label_encoder"] = joblib.load("label_encoder.pkl")
    artifacts["feature_columns"] = joblib.load("models/feature_columns.pkl")
    artifacts["scaler"] = joblib.load("scaler.pkl")
    artifacts["pca"] = joblib.load("pca_bert_embeddings.pkl")
    # SentenceTransformer can take a moment to download; caching helps
    artifacts["sbert_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    return artifacts

art = load_artifacts()
VOTING_CLF = art["voting_clf"]
LABEL_ENCODER = art["label_encoder"]
FEATURE_COLUMNS = art["feature_columns"]
SCALER = art["scaler"]
PCA = art["pca"]
SBERT = art["sbert_model"]

# ----------------------------- Keyword lists -----------------------------
skills_keywords = [
    "python","java","c++","c#","javascript","typescript","go","ruby","r","php","matlab",
    "react","angular","vue","django","flask","spring","nodejs","express","html","css",
    "sql","mysql","postgresql","oracle","mongodb","redis","cassandra","nosql","hive","spark",
    "aws","azure","gcp","docker","kubernetes","terraform","ansible","jenkins","ci/cd","devops",
    "pandas","numpy","scikit-learn","tensorflow","pytorch","keras","matplotlib","seaborn",
    "nlp","machine learning","deep learning","data science","big data","hadoop",
    "excel","tableau","powerbi","sas","jira","confluence","salesforce","workday","peoplesoft",
    "penetration testing","ethical hacking","cybersecurity","firewall","encryption","vpn","siem"
]
soft_skills_keywords = [
    "leadership","communication","teamwork","problem solving","analytical","management","adaptability",
    "collaboration","creativity","critical thinking","negotiation","decision making",
    "organization","time management","mentoring"
]

# ----------------------------- Helpers -----------------------------
def read_resume_bytes(uploaded_file):
    """Read uploaded file bytes and return text."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            return text
        elif ext == ".docx":
            uploaded_file.seek(0)
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            uploaded_file.seek(0)
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            return ""
    except Exception as e:
        st.warning(f"Cannot read {uploaded_file.name}: {e}")
        return ""

def extract_features(text):
    words = text.split()
    features = {"char_count": len(text), "word_count": len(words)}
    
    text_lower = text.lower()
    skills_found = []
    
    for kw in skills_keywords + soft_skills_keywords:
        # match as a whole word only
        if re.search(rf'\b{re.escape(kw.lower())}\b', text_lower):
            skills_found.append(kw)
    
    # order skills by decreasing mentions
    uniq_sorted = sorted(set(skills_found), key=lambda k: -len(re.findall(rf'\b{re.escape(k.lower())}\b', text_lower)))
    features["skills_list"] = ", ".join(uniq_sorted)
    
    # ensure features expected by model are present
    for col in FEATURE_COLUMNS:
        features.setdefault(col, 0)
    cols = list(FEATURE_COLUMNS) + ["skills_list"]
    return pd.DataFrame([features])[cols]

def get_reduced_bert(text):
    try:
        bert_vector = SBERT.encode([text], show_progress_bar=False)
        reduced = PCA.transform(bert_vector)
        return reduced
    except Exception as e:
        st.warning(f"BERT/PCA failed: {e}")
        # fallback zeros with PCA dimensionality if available
        d = PCA.n_components_ if hasattr(PCA, "n_components_") else 34
        return np.zeros((1, d))

def generate_summary(text, max_sentences=4):
    sentences = [s.strip() for s in re.split(r"\n|\.", text) if len(s.strip()) > 20]
    return " ".join(sentences[:max_sentences]) if sentences else text[:300]

# ----------------------------- Main UI -----------------------------
st.title("🚀 ExcelR Resume Dashboard")
st.markdown("Upload one or more resumes. This app performs aggregated EDA, per-resume insights, visualizations and optional SHAP explanations.")

uploaded_files = st.file_uploader("Upload resumes (PDF / DOCX / TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload resumes to start.")
    st.stop()

if len(uploaded_files) > max_resumes:
    st.warning(f"Limiting to the first {max_resumes} resumes.")
    uploaded_files = uploaded_files[:max_resumes]

# ----------------------------- Process each resume -----------------------------
res_list = []
agg_skill_counts = {}
class_counts = {}
confidence_records = []

with st.spinner("Parsing and extracting features..."):
    for f in uploaded_files:
        text = read_resume_bytes(f)
        if not text or len(text.strip()) == 0:
            st.warning(f"{f.name} appears empty or unreadable — skipped.")
            continue

        features_df = extract_features(text)
        skills_list = features_df["skills_list"].iloc[0]
        tech_skills = [k for k in skills_keywords if k.lower() in skills_list.lower()]
        soft_skills = [k for k in soft_skills_keywords if k.lower() in skills_list.lower()]

        # scale structured numeric features (safe)
        struct_cols = list(FEATURE_COLUMNS)[:16] if len(FEATURE_COLUMNS) >= 16 else list(FEATURE_COLUMNS)
        try:
            structured_scaled = SCALER.transform(features_df[struct_cols].values)
        except Exception:
            structured_scaled = features_df[struct_cols].values

        # BERT reduced embeddings
        bert_emb = get_reduced_bert(text)  # shape (1, d)
        # ensure bert_emb is 2D
        if bert_emb.ndim == 1:
            bert_emb = bert_emb.reshape(1, -1)

        # combine
        try:
            X_final = np.hstack([structured_scaled, bert_emb])
        except Exception:
            # shape mismatches: pad with zeros for safety
            d_bert = bert_emb.shape[1]
            pad = np.zeros((structured_scaled.shape[0], max(0, d_bert - (X_final.shape[1] if 'X_final' in locals() else 0))))
            X_final = np.hstack([structured_scaled, bert_emb])

        # predict
        try:
            pred = VOTING_CLF.predict(X_final)[0]
            proba = VOTING_CLF.predict_proba(X_final)[0]
            label = LABEL_ENCODER.inverse_transform([pred])[0]
            top_prob = float(np.max(proba))
        except Exception:
            label = "unknown"
            top_prob = 0.0
            proba = np.zeros(len(LABEL_ENCODER.classes_))

        # aggregate skills (whole-word match)
        for s in tech_skills + soft_skills:
            c = len(re.findall(rf'\b{re.escape(s.lower())}\b', text.lower()))
            agg_skill_counts[s] = agg_skill_counts.get(s, 0) + c


        class_counts[label] = class_counts.get(label, 0) + 1
        confidence_records.append({"file": f.name, "pred": label, "confidence": top_prob})

        res_list.append({
            "file": f.name,
            "text": text,
            "skills_list": skills_list,
            "tech_skills": tech_skills,
            "soft_skills": soft_skills,
            "pred": label,
            "proba": proba,
            "top_prob": top_prob,
            "X_final": X_final,
            "features_df": features_df
        })

# ----------------------------- Aggregated Overview -----------------------------
st.header("📊 Aggregate Overview")
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    st.subheader("Resume Count")
    st.metric("Uploaded", len(res_list))

with col2:
    st.subheader("Top Predicted Classes")
    if class_counts:
        class_df = pd.DataFrame({"class": list(class_counts.keys()), "count": list(class_counts.values())}).sort_values("count", ascending=False)
        fig = px.bar(class_df, x="class", y="count", text="count", template=CHART_TEMPLATE)
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Number of Resumes", height=260)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No class predictions available yet.")

with col3:
    st.subheader("Confidence Distribution")
    conf_df = pd.DataFrame(confidence_records)
    if not conf_df.empty:
        fig = px.histogram(conf_df, x="confidence", nbins=10, template=CHART_TEMPLATE, title="Top-probability distribution")
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No confidence data")

# ----------------------------- Aggregated Skills EDA -----------------------------
st.header("🔎 Aggregate Skills EDA")
if agg_skill_counts:
    agg_df = pd.DataFrame({"skill": list(agg_skill_counts.keys()), "mentions": list(agg_skill_counts.values())})
    agg_df = agg_df.sort_values("mentions", ascending=False).reset_index(drop=True)

    top_k = st.slider("Top K skills to show", 5, min(30, max(5, len(agg_df))), value=12)
    st.markdown("### 🔝 Top Skills Across Uploaded Resumes")
    st.dataframe(agg_df.head(top_k).assign(**{"Share (%)": (agg_df["mentions"] / agg_df["mentions"].sum() * 100).round(1)}))

    csv_bytes = agg_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download aggregated skills CSV", csv_bytes, file_name="aggregated_skills.csv", mime="text/csv")

    fig_bar = px.bar(agg_df.head(top_k), x="mentions", y="skill", orientation="h", text="mentions", template=CHART_TEMPLATE, title="Top skill mentions (aggregated)")
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### ☁️ Word Cloud (Aggregated)")
    wc_text = " ".join([r["text"] for r in res_list])
    wc = WordCloud(width=800, height=300, background_color=WC_BG, colormap="tab10").generate(wc_text)
    plt.figure(figsize=(10, 4)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off"); st.pyplot(plt); plt.close()

    st.markdown("### 🌐 Skill Co-occurrence Network (Aggregated)")
    from collections import Counter
    cooc = Counter()
    for r in res_list:
        skills_set = set(r["tech_skills"] + r["soft_skills"])
        for s1 in skills_set:
            for s2 in skills_set:
                if s1 < s2:
                    cooc[(s1, s2)] += 1
    G = nx.Graph()
    for (a, b), w in cooc.items():
        if w >= 1:
            G.add_edge(a, b, weight=w)
    if G.number_of_nodes() == 0:
        st.info("Not enough skill overlap to build network.")
    else:
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for e in G.edges():
            x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
        node_x, node_y, node_text, node_size = [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x); node_y.append(y); node_text.append(n)
            node_size.append(8 + (agg_skill_counts.get(n, 1) / max(agg_skill_counts.values())) * 30)
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
                                marker=dict(size=node_size, color='lightskyblue'))
        fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, template=CHART_TEMPLATE, height=450))
        st.plotly_chart(fig_net, use_container_width=True)
else:
    st.info("No skills found across uploaded resumes.")

# ----------------------------- Extra NLP Metrics & N-Grams -----------------------------
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords

st.header("🧠 Advanced NLP EDA")

all_text = " ".join([r["text"] for r in res_list])

# Compute readability metrics
def readability_scores(text):
    words = text.split()
    sents = max(1, text.count("."))
    avg_words = len(words) / sents
    syllables = sum([len([ch for ch in w if ch.lower() in "aeiou"]) for w in words])
    flesch = 206.835 - 1.015*avg_words - 84.6*(syllables/len(words))
    fog = 0.4 * (avg_words + 100*(sum(len(w)>6 for w in words)/len(words)))
    return flesch, fog

flesch, fog = readability_scores(all_text)
st.metric("Flesch Reading Ease", round(flesch,1))
st.metric("Gunning Fog Index", round(fog,1))

# Diversity metrics
unique_words = len(set(all_text.split()))
st.metric("Unique Word %", round(100*unique_words/len(all_text.split()),1))
st.metric("Skill Density", round(100*len(agg_skill_counts)/len(all_text.split()),2))

# N-Gram Analysis
st.subheader("📑 N-Gram Analysis (Top Words/Phrases)")
n = st.selectbox("Select n-gram size", [1,2,3], index=0)

vectorizer = CountVectorizer(ngram_range=(n,n), stop_words=stopwords.words("english")).fit([all_text])
ngram_counts = vectorizer.transform([all_text])
ngram_df = pd.DataFrame({
    "ngram": vectorizer.get_feature_names_out(),
    "count": ngram_counts.toarray()[0]
}).sort_values("count", ascending=False).head(20)

fig_ng = px.bar(ngram_df, x="count", y="ngram", orientation="h",
                title=f"Top {n}-grams across all resumes",
                template=CHART_TEMPLATE)
fig_ng.update_layout(height=420)
st.plotly_chart(fig_ng, use_container_width=True)

# TF-IDF keywords
st.subheader("🌟 TF-IDF Weighted Terms (Unique Strengths)")
tfidf = TfidfVectorizer(max_features=30, stop_words=stopwords.words("english"))
tfidf_fit = tfidf.fit_transform([all_text])
tfidf_df = pd.DataFrame({"term": tfidf.get_feature_names_out(), "score": tfidf_fit.toarray()[0]})
tfidf_df = tfidf_df.sort_values("score", ascending=False)
fig_tf = px.bar(tfidf_df.head(15), x="score", y="term", orientation="h", template=CHART_TEMPLATE,
                title="Top Weighted TF-IDF Terms")
st.plotly_chart(fig_tf, use_container_width=True)

# ----------------------------- Per-resume detail -----------------------------
st.header("🧾 Per-resume detail (expand each item)")

for idx, r in enumerate(res_list):
    with st.expander(f"{r['file']} — Pred: {r['pred']} ({r['top_prob']*100:.1f}%)"):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("**Preview / Summary**")
            summary = generate_summary(r["text"], max_sentences=3)
            st.write(summary)
        with c2:
            st.metric("Words", len(r["text"].split()))
            st.metric("Detected skills", len((r["tech_skills"] + r["soft_skills"])))

        # badges
        if r["skills_list"]:
            skills = [s for s in r["skills_list"].split(", ") if s]
            badge_html = "<div style='display:flex;flex-wrap:wrap;align-items:center;'>"
            palette = px.colors.qualitative.Dark24
            for i, s in enumerate(skills):
                color = palette[i % len(palette)]
                cnt = len(re.findall(rf'\b{re.escape(s.lower())}\b', r["text"].lower()))
                badge_html += f"<div style='margin:6px;padding:6px 10px;border-radius:10px;background:{color};color:white;font-weight:700;'>{s.title()} <span style='opacity:0.9'>&nbsp;({cnt})</span></div>"

            badge_html += "</div>"
            components.html(badge_html, height=110)
        else:
            st.info("No skills detected in this resume.")

        # per-resume charts
        st.markdown("**Per-resume skill breakdown**")
        counts = {s: len(re.findall(rf'\b{re.escape(s.lower())}\b', r["text"].lower()))
                  for s in (r["tech_skills"] + r["soft_skills"])}

        if counts:
            sdf = pd.DataFrame({"skill": list(counts.keys()), "count": list(counts.values())}).sort_values("count", ascending=False)
            fig1 = px.pie(sdf, names="skill", values="count", title="Skill share (this resume)", hole=0.4, template=CHART_TEMPLATE)
            fig2 = px.bar(sdf, x="count", y="skill", orientation="h", title="Skill freq (this resume)", template=CHART_TEMPLATE)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)

        # resume wordcloud
        st.markdown("**Resume Word Cloud**")
        wc = WordCloud(width=600, height=250, background_color=WC_BG, colormap="tab10").generate(r["text"])
        plt.figure(figsize=(8, 3)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off"); st.pyplot(plt); plt.close()

        # class probabilities
        st.markdown("**Top Class Probabilities**")
        try:
            proba = r["proba"]
            proba_df = pd.DataFrame({"Class": LABEL_ENCODER.classes_, "Probability": proba * 100}).sort_values("Probability", ascending=True)
            fig_conf = px.bar(proba_df, x="Probability", y="Class", orientation="h", text="Probability", template=CHART_TEMPLATE)
            fig_conf.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(fig_conf, use_container_width=True)
        except Exception:
            st.info("Probability information not available.")

        # soft-skill radar (heuristic)
        st.markdown("**Soft-skill radar (heuristic)**")
        traits = ["Leadership", "Communication", "Teamwork", "Analytical", "Creativity", "Adaptability"]
        scores = [min(100, max(5, r["text"].lower().count(t.lower()) * 12)) for t in traits]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=scores, theta=traits, fill='toself'))
        fig_r.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=False, template=CHART_TEMPLATE, height=350)

        # ⚡ Add unique key here using the resume index `idx`
        st.plotly_chart(fig_r, use_container_width=True, key=f"radar_{idx}")

        # SHAP (best-effort)
        if show_shap:
            st.markdown("### 🧠 SHAP Explanation (best-effort; may be slow)")
            if not HAS_SHAP:
                st.info("SHAP is not installed in this environment. Install `shap` to enable explanations.")
            else:
                try:
                    # Build background using random rows from all X_final if available
                    # For safety use small sample (shap_samples)
                    all_X = np.vstack([ri["X_final"] for ri in res_list if "X_final" in ri])
                    if all_X.shape[0] > shap_samples:
                        background = all_X[np.random.choice(all_X.shape[0], shap_samples, replace=False)]
                    else:
                        background = all_X
                    # Use KernelExplainer on predict_proba to get class-level explanations
                    # KernelExplainer is slow — warn the user and limit to single instance
                    with st.spinner("Computing SHAP values (this may take a while)..."):
                        explainer = shap.KernelExplainer(VOTING_CLF.predict_proba, background)
                        shap_values = explainer.shap_values(r["X_final"], nsamples=shap_samples)
                        # shap_values is list (one per class). Show summary for top predicted class
                        top_class_idx = int(np.argmax(r["proba"]))
                        st.markdown(f"**Top predicted class:** {LABEL_ENCODER.classes_[top_class_idx]}")
                        # Plot SHAP bar for that class
                        plt.figure(figsize=(8, 3))
                        shap.summary_plot(shap_values[top_class_idx], r["X_final"], feature_names=list(FEATURE_COLUMNS) + [f"bert_{i}" for i in range(r["X_final"].shape[1] - len(FEATURE_COLUMNS))], show=False, plot_type="bar")
                        st.pyplot(plt); plt.close()
                except Exception as e:
                    st.warning(f"SHAP failed or too slow for this resume: {e}")

# ----------------------------- Top-skill Trend Animation -----------------------------
st.header("📈 Top-skill Trend (by upload order)")
if agg_skill_counts and 'agg_df' in locals():
    top_skills = list(agg_df["skill"].head(6))
    frames = []
    for idx, r in enumerate(res_list):
        row = {"resume_idx": idx, "resume": r["file"]}
        for s in top_skills:
            row[s] = len(re.findall(rf'\b{re.escape(s.lower())}\b', r["text"].lower()))
        frames.append(row)
    trend_df = pd.DataFrame(frames).fillna(0)
    fig_anim = px.line(trend_df, x="resume_idx", y=top_skills, markers=True, template=CHART_TEMPLATE,
                       labels={"resume_idx": "Resume index (order uploaded)"}, title="Top skills across uploaded resumes")
    fig_anim.update_layout(height=420)
    st.plotly_chart(fig_anim, use_container_width=True)

# ----------------------------- Exports & Final tips -----------------------------
st.markdown("---")
st.markdown("### ✅ Actions")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Export per-resume CSVs"):
        zip_buffer = BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for r in res_list:
                df = pd.DataFrame(list({s: r["text"].lower().count(s.lower()) for s in (r["tech_skills"] + r["soft_skills"])}).items(), columns=["skill", "mentions"])
                csvb = df.to_csv(index=False).encode("utf-8")
                zf.writestr(f"{r['file']}_skills.csv", csvb)
        zip_buffer.seek(0)
        st.download_button("📥 Download ZIP (per-resume skills)", zip_buffer, file_name="per_resume_skills.zip", mime="application/zip")
with c2:
    if 'agg_df' in locals() and st.button("Export aggregated report (CSV)"):
        buf = BytesIO()
        agg_df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("📥 Download Aggregated Skills Report", buf, file_name="aggregated_skills.csv", mime="text/csv")
with c3:
    st.markdown("#### 💡 Tips")
    st.markdown("- Hover charts for details, zoom and pan supported. \n- Toggle SHAP only if needed (it can be slow). \n- Use 'Top K' slider to tune aggregated view.")

st.markdown("App generated by Excelr Resume Dashboard — tweak `FEATURE_COLUMNS` and model paths as required.")
