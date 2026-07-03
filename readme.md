<h1 align="center">☕🗺️ Geo-SentimentCafeAnalysis</h1>

<p align="center"><b>Analisis Geo-Sentimen & Pemodelan Topik Ulasan Kafe di Purwokerto</b><br/>
<i>Geo-sentiment analysis and topic modeling of café reviews — a spatial comparison of Education vs. Tourism zones in Purwokerto.</i></p>

<p align="center">
  <a href="https://colab.research.google.com/github/JordanCodeGit/Geo-SentimentCafeAnalysis/blob/main/Analisis_Geo_Sentimen_Kafe_Purwokerto_Kelompok_4.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/IndoBERT-auto--labeling-4285F4?style=flat" alt="IndoBERT"/>
  <img src="https://img.shields.io/badge/Folium-geospatial-77B829?style=flat&logo=leaflet&logoColor=white" alt="Folium"/>
  <img src="https://img.shields.io/badge/F1--Macro-0.729-brightgreen?style=flat" alt="F1 0.729"/>
</p>

---

## 📌 Overview

Purwokerto has two distinct commercial characters: an **Education Zone** around Telkom University, Unsoed, and UMP dominated by students and campus life, and a **Tourism Zone** in Baturraden built around the natural draw of Mount Slamet. This project asks a simple question with a spatial twist:

> **Does the *location* of a café change what customers complain about — and how happy they are?**

To answer it, the notebook scrapes thousands of Google Maps reviews across **20 cafés**, auto-labels historical sentiment with **IndoBERT**, trains a classical ML classifier under a strict **out-of-time validation** scheme, then projects sentiment onto an interactive **geospatial dashboard** and mines the root causes of negative reviews with topic modeling and n-gram analysis.

This is a **Natural Language Processing (NLP)** course project at **Telkom University, Kampus Purwokerto** (Course code `CAK4NBB3-PSIIF-10-REG01`, semester 2526/2).

---

## 🎯 Business Questions

1. How does the **sentiment distribution** (positive / neutral / negative) of café reviews compare between the **Education Zone** and the **Tourism Zone (Baturraden)**?
2. What **topics** dominate customer discourse in each zone?
3. Is there a measurable **spatial pattern** in review sentiment across Purwokerto?
4. Which factors most influence customer satisfaction in each zone?

### 🧪 Research Hypotheses
- Complaint topics differ by geography — the Education Zone is dominated by **infrastructure** issues (parking, Wi-Fi/connectivity), while the Tourism Zone is dominated by **accessibility** issues (road conditions, congestion).
- Negative reviews in the Tourism Zone center more on a **price-vs-experience mismatch** than in the Education Zone.

---

## 🗂️ Dataset

Reviews are scraped **live** from **Google Maps** using **Apify** (`compass/crawler-google-places`), Indonesian language, sorted newest-first, with original (untranslated) text to preserve local slang.

| Attribute | Value |
|-----------|-------|
| Source | Google Maps Reviews (via Apify) |
| Targets | **20 cafés** — 10 Education Zone + 10 Tourism Zone (Baturraden) |
| Raw reviews collected | **5,755** valid text reviews |
| Per-zone volume | Education **2,973** · Tourism **2,782** |
| Captured metadata | café name, zone, GPS lat/lon, global rating, reviewer, star rating, timestamp, review text |

**Café roster**

| Tourism Zone (Baturraden) | Education Zone (Campus / City) |
|---------------------------|--------------------------------|
| Ebony Cafe · MASSAPI · Penak Mawon · Cerita Alam Resto · Taman Langit · New Kalarasa Resto & Coffee · Layana Kopi · Lembah Patih · Polaris Coffee & Resto · Kopi Kami untuk Kamu | Arasta Alpha Overste Isdiman · Society Coffee House · Kopi Arasta Hotel Wisata Niaga · Praketa · At Cafe Campus · Lav Cafe · Elskoffie · Alas House · Mimosa Coffee Kitchen & Dessert · et al Coffee |

> ⚠️ The raw dataset is generated at runtime — no CSV is committed. Reproducing the scrape requires an **Apify API token** stored in Colab Secrets as `APIFY_API_TOKEN`.

---

## 🔬 Methodology (CRISP-DM)

The pipeline is organized into four phases, following a CRISP-DM flow with a deliberate **temporal (out-of-time) split** to guarantee honest evaluation.

### 🟦 Phase I — Data Engineering & Temporal Separation
- **Live scraping** of 20 cafés into a unified raw dataset with spatial coordinates.
- **Chronological Bifurcation (Out-of-Time split):** history (**≤ 2025**, 4,481 rows) becomes the training universe; **2026** (1,274 rows) is quarantined as an unseen production set. This prevents the model from ever "seeing the future."

### 🟩 Phase II — Historical Modeling
- **Text cleaning blueprint:** case folding → cleansing → **slang normalization** → stopword removal → **Sastrawi stemming**. Negation words (`tidak`, `bukan`, `belum`, `kurang`, …) are deliberately **protected** from removal so sentiment isn't flipped.
- **IndoBERT auto-labeling:** `mdhugol/indonesia-bert-sentiment-classification` acts as a *teacher model*, generating ground-truth labels for the historical set (a zero-shot knowledge-distillation step).
- **Feature extraction:** TF-IDF with unigrams + bigrams, top 5,000 features → `(4,468 × 5,000)` matrix, saved as `tfidf_vectorizer.pkl`.
- **Class balancing:** `RandomUnderSampler` levels the classes to **169 each (507 total)** for a fair, unbiased benchmark.
- **Model tournament:** `GridSearchCV` (5-fold Stratified, `f1_macro`) over **Logistic Regression, Linear SVC, Complement NB, and Random Forest**.

### 🟨 Phase III — Live Production Inference (2026)
- The winning model + vectorizer are loaded and applied **blind** to the 2026 reviews (using `.transform`, never `.fit` — no leakage).
- Autonomous prediction of sentiment on **1,272** unseen reviews → `Dataset_Production_2026_Predicted.csv`.

### 🟧 Phase IV — Downstream Analytics & Spatial Dashboard
- **LDA topic modeling** on negative reviews to surface problem clusters.
- **Negative-bigram mining** (with domain stopword + mixed-sentiment filtering) to isolate operational pain points per zone.
- **Interactive Folium dashboard** with dual temporal layers (historical ground truth vs. 2026 prediction), sentiment-colored pins, and a complaint-intensity heat radius.
- **Time-filtered widget** (year / month / week dropdowns) and a **high-resolution static spatial map** for print.

---

## 🏆 Model Results

5-fold Stratified Cross-Validation (F1-Macro) on the balanced 507-sample historical set, after hyperparameter tuning:

| Model | Val F1-Macro | Best Params |
|-------|:------------:|-------------|
| 🥇 **Logistic Regression** | **0.7292** | `C=0.1, solver=liblinear` |
| Linear SVC | 0.7261 | `C=0.1, loss=squared_hinge` |
| Random Forest | 0.7056 | `max_depth=50, n_estimators=200` |
| Complement NB | 0.6856 | `alpha=2.0, norm=True` |

**Logistic Regression** was selected and retrained on the full balanced set. The strong regularization (`C=0.1`) indicates a model that generalizes rather than overfits.

---

## 🔎 Key Findings

**1. Blind 2026 sentiment distribution** (1,272 reviews)

| Sentiment | Count | Share |
|-----------|:-----:|:-----:|
| 🟢 Positive | 742 | 58.3% |
| 🟠 Neutral | 271 | 21.3% |
| 🔴 Negative | 259 | 20.4% |

**2. Spatial disparity** — the two zones do *not* behave the same:

| Zone | Negative-review ratio |
|------|:---------------------:|
| 🎓 Education | **28.8%** |
| 🏞️ Tourism (Baturraden) | **15.5%** |

Students appear **less tolerant of operational inefficiency** than tourists.

**3. Root cause of complaints** — n-gram mining reveals the biggest issue is **not the product**, but a **service bottleneck**. Dominant negative phrases: *"layan lama"* (slow service), *"makan lama"*, *"nunggu lama"* (long waits), and friction over basics like *"tukang parkir"* (parking). This aligns with the infrastructure/operational hypothesis for the Education Zone.

---

## 🛠️ Tech Stack

**Language & Environment** — Python · Google Colab

**Data Collection** — `apify-client` (Google Maps `compass/crawler-google-places`)

**NLP** — Sastrawi (Indonesian stemmer) · NLTK (stopwords, tokenizer) · Hugging Face **Transformers** + PyTorch (IndoBERT teacher model)

**Modeling** — scikit-learn (`TfidfVectorizer`, `CountVectorizer`, `LogisticRegression`, `LinearSVC`, `ComplementNB`, `RandomForestClassifier`, `GridSearchCV`, `StratifiedKFold`, `LatentDirichletAllocation`) · imbalanced-learn (`RandomUnderSampler`)

**Visualization** — Folium (interactive geospatial) · Matplotlib · Seaborn · ipywidgets (temporal filtering)

---

## 📦 Generated Artifacts

Running the notebook end-to-end produces (and bundles into a downloadable ZIP):

| File | Description |
|------|-------------|
| `model_sentimen_terbaik.pkl` | Trained Logistic Regression classifier |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `Dataset_Production_2026_Predicted.csv` | 2026 reviews with AI-predicted sentiment |
| `Confusion_Matrix_Komparasi.png` | Cross-validated confusion matrices for all 4 models |
| `Visualisasi_Prediksi_2026_Fixed.png` | 2026 sentiment donut + bar chart |
| `Komparasi_Sentimen_Regional_2026.png` | 100% stacked zone comparison |
| `Akar_Masalah_Operasional_2026_Fixed.png` | Top negative bigrams (root causes) |
| `Peta_Spasial_Statis_2026.png` | High-res static spatial map |
| `Dashboard_GeoSentiment_Purwokerto_Final.html` | Interactive dual-temporal Folium dashboard |

---

## 🚀 Getting Started

### Run in the cloud (recommended)
Click the **Open In Colab** badge above and **Runtime → Run all**.

**Before running the scraping cells**, add your Apify token via the 🔑 **Secrets** panel in Colab:
```
Name:  APIFY_API_TOKEN
Value: <your Apify API token>
```

### Run locally

```bash
git clone https://github.com/JordanCodeGit/Geo-SentimentCafeAnalysis.git
cd Geo-SentimentCafeAnalysis

pip install apify-client Sastrawi imbalanced-learn folium \
            transformers torch scikit-learn nltk pandas numpy matplotlib seaborn ipywidgets

jupyter notebook Analisis_Geo_Sentimen_Kafe_Purwokerto_Kelompok_4.ipynb
```

> 💡 The IndoBERT auto-labeling step downloads a ~500 MB model on first run and benefits from a GPU runtime.

---

## 📁 Repository Structure

```
Geo-SentimentCafeAnalysis/
├── Analisis_Geo_Sentimen_Kafe_Purwokerto_Kelompok_4.ipynb   # Full 4-phase pipeline
└── readme.md                                                # This file
```

The notebook flows through **24 cells** across four phases: data engineering & temporal split → historical IndoBERT labeling + model tournament → blind 2026 inference → topic modeling, n-gram root-cause analysis, and geospatial dashboards.

---

## 👥 Authors

| Name | Student ID |
|------|-----------|
| **Jordan Angkawijaya** | 2311102139 |
| **Naswa Malika Putri** | 2311102232 |

**Kelompok 4** — Natural Language Processing, S1 Teknik Informatika, Fakultas Informatika, **Telkom University — Kampus Purwokerto**.

---

## 📚 References

1. Oraby, S., et al. — Sarcasm Corpus V2 methodology *(companion project reference)*.
2. `mdhugol/indonesia-bert-sentiment-classification` — IndoBERT sentiment model, Hugging Face Hub.
3. Apify — `compass/crawler-google-places` actor documentation.
