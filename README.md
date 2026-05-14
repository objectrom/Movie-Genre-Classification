# Movie-Genre-Classification

# 🎬 Multi-Label Movie Genre Classification
### DSS DeCal · UC Berkeley

> Predicting movie genres from overview text using transformer-based embeddings and multi-label classification.

---

## 📌 Overview

This project explores whether a movie's **plot overview text alone** can predict its genre(s). Using NLP embeddings from state-of-the-art transformer models, we built a multi-label classifier across 14 genres on a dataset of tens of thousands of films spanning several decades.

**Key finding:** Logistic Regression on high-dimensional transformer embeddings outperforms gradient boosting models (XGBoost, LightGBM) — because genre boundaries are already approximately linear in embedding space.

---

## 📂 Repository Structure

```
movie-genre-classification/
├── README.md
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_embeddings.ipynb
│   └── 04_modeling.ipynb
├── presentation/
│   └── slides.pptx
├── data/
│   └── README.md          ← dataset download instructions
└── .gitignore
```

---

## 🗂️ Dataset

- IMDb movie metadata with thousands of films
- Fields used: `overview` (primary NLP feature), `genres` (multi-label target), `budget`, `revenue`, `release_date`, `title`, `tagline`
- Multi-label genre tags enabled multi-hot classification

---

## ⚙️ Methodology

### Preprocessing Pipeline
- Parsed semi-structured IMDb fields using regex (`re.findall`)
- Extracted structured features (genres, languages, companies, countries)
- Standardized `release_date` using `pd.to_datetime()`
- Dropped rows with missing overview text
- Applied **MultiLabelBinarizer** for multi-hot genre encoding

### Genre Imbalance Treatment

| Strategy | Result |
|----------|--------|
| Genre frequency cutoff (<2,000 samples) | Reduced from 20 → 14 genres ✅ |
| Class weight rebalancing | Over-weighted rare genres hurt dominant class performance ❌ |
| Cutoff + rebalancing combined | No additional benefit over cutoff alone ❌ |

**→ Genre cutoff at 2,000 samples was the most effective strategy.**

### Embeddings

Converted movie overviews to dense semantic vectors using SentenceTransformer models:

| Model | Notes |
|-------|-------|
| `all-mpnet-base-v2` | Baseline |
| `all-MiniLM-L6-v2` | Faster, but less semantic depth |
| `intfloat/e5-large-v2` ✅ | Best — captures storyline, tone, and theme; 1,024-dim vectors |

Best config: **title + tagline + overview** combined embedding (excluding date features)

### Feature Engineering
- Log-transformed `budget` and `revenue` (right-skewed distributions)
- Combined semantic text embeddings with structured metadata

---

## 📊 Results

### Best Configuration
```
e5-large-v2 + cutoff 3,000 + threshold 0.28
+ title + tagline + overview embedding
+ OneVsRest Logistic Regression
```

| Metric | Score |
|--------|-------|
| **Micro F1** | **0.6825** |
| **Macro F1** | **0.6608** |

### Model Comparison
*(MiniLM-L6-v2 / Cutoff 2,000 / Threshold 0.25)*

| Model | Macro F1 | Micro F1 | Train Time |
|-------|----------|----------|------------|
| **Logistic Regression** ✅ | **0.647** | **0.683** | ~1 min |
| LightGBM | 0.525 | 0.593 | ~6 min |
| XGBoost | 0.528 | 0.597 | ~12 min |

### Per-Genre Performance

| Genre | F1 | Notes |
|-------|----|-------|
| Documentary | 0.795 | Distinctive vocabulary makes it easy to separate |
| Horror | 0.744 | Strong tonal language in overviews |
| Drama | 0.738 | Largest class; slight over-prediction |
| **Adventure** | **0.502** | Hardest — heavily overlaps with Action, Sci-Fi, Drama |

### Why Logistic Regression Won

`e5-large-v2` produces **1,024-dimensional vectors** where genre clusters are already linearly separable. Tree-based models (XGBoost, LightGBM) split one feature at a time and need exponentially more splits to approximate what logistic regression captures in a single hyperplane — resulting in higher compute cost with no accuracy gain.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-e5--large--v2-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LogisticRegression-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-compared-red)
![LightGBM](https://img.shields.io/badge/LightGBM-compared-brightgreen)

```
sentence-transformers · scikit-learn · pandas · numpy · matplotlib · seaborn · xgboost · lightgbm
```

---

## 🔭 Future Work

**Data**
- Scrape additional metadata (director, cast, studio) for stronger genre signals
- Augment rare-genre samples via paraphrase generation (Adventure, Crime, Romance)

**Embeddings**
- Test newer models: `GTE-large`, `BGE-M3`, `text-embedding-3-large`
- Fine-tune `e5-large-v2` on movie overviews with contrastive genre labels → target Macro F1 > 0.72
- Ensemble of separate embedding towers (title / tagline / overview)

**Modeling**
- Multi-label neural classifiers (BERT + classification head) for genre co-occurrence modeling
- Graph-based label propagation (Drama+Romance frequently co-occur)
- Per-genre threshold tuning (Adventure may benefit from lower cutoff ~0.20)
