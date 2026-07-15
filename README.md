# PCOS Diagnostic Tool

A two-stage clinical decision-support web app for Polycystic Ovary Syndrome (PCOS) detection and phenotype classification, built with Streamlit.

🔗 **Live app:** https://pcos-phenotype-classfication.streamlit.app/

---

## Overview

This tool applies the **Rotterdam 2003 criteria** to determine PCOS presence, then uses a trained **Random Forest** model to classify the specific PCOS phenotype (A–D) based on 26 clinically selected features. Results are paired with SHAP-based explainability so predictions aren't a black box.

**Two-stage pipeline:**

1. **PCOS Detection (Rule-Based)** — Applies the Rotterdam criteria (oligo/anovulation, hyperandrogenism, polycystic ovarian morphology) to the entered clinical data. At least 2 of 3 criteria must be met for a positive result.
2. **Phenotype Classification (ML)** — A Random Forest pipeline with a custom TOMIM (Top Mutual-Information) feature selector predicts the phenotype (A, B, C, or D) with per-class probability scores, then explains the prediction using SHAP TreeExplainer.

## The Four PCOS Phenotypes

| Phenotype | Criteria Met | Description |
|---|---|---|
| **A** | Anovulation + Hyperandrogenism + PCOM | Full Classic PCOS — highest metabolic risk |
| **B** | Anovulation + Hyperandrogenism | Classic without polycystic ovaries |
| **C** | Hyperandrogenism + PCOM | Ovulatory PCOS — cycles remain regular |
| **D** | Anovulation + PCOM | Non-Androgenic PCOS — lowest metabolic risk |

## Features

- Six-section guided intake form (anthropometrics, vitals, menstrual/reproductive history, labs, ultrasound, symptoms)
- Rotterdam rule-based PCOS screening
- ML-based phenotype classification with confidence scores
- Interactive clinical dashboard with Plotly visualizations
- SHAP-based feature importance / explainability per prediction
- Custom-styled UI (`style.css`)

## Tech Stack

- **Frontend/App:** Streamlit
- **ML:** scikit-learn, imbalanced-learn, XGBoost, SHAP
- **Data:** NumPy, Pandas
- **Visualization:** Plotly

## Project Structure

```
pcos-app/
├── pcos_app.py              # Main Streamlit application
├── style.css                # Custom UI styling
├── requirements.txt         # Python dependencies
├── p2_model.pkl             # Trained phenotype classification model (Stage 2)
├── ovarian_morphology.jpg   # Reference image — normal vs. polycystic ovaries
├── phenotypes.jpg           # Reference image — phenotype classification chart
├── P1_Pipeline.ipynb        # Notebook: PCOS detection model training
├── P2_Pipeline.ipynb        # Notebook: Phenotype classification model training
├── .streamlit/              # Streamlit config
├── .devcontainer/           # Dev container config
└── scripts/                 # Auxiliary/debug scripts (not used by the live app)
    └── debug_pipeline.py
```

## Running Locally

```bash
git clone https://github.com/Yeonskx/pcos-app.git
cd pcos-app
pip install -r requirements.txt
streamlit run pcos_app.py
```

The app will open at `http://localhost:8501`.

## Deployment

This app is deployed via **Streamlit Community Cloud**, which auto-redeploys on every push to `main`. Streamlit Cloud reads `requirements.txt` from the repo root to build the environment — if you add a new import to `pcos_app.py`, make sure the corresponding package is added there too.

## Model Details

- **Feature Selection:** Custom `TOMIMSelector` (Top mutual-information features) selects the most informative features via `mutual_info_classif`.
- **Model:** Random Forest classifier trained on 26 clinical features (anthropometrics, labs, ultrasound follicle counts, symptoms).
- **Explainability:** SHAP TreeExplainer values are computed per-prediction to surface the top features driving each phenotype classification.

## Disclaimer

⚠️ This tool is intended for **research and educational purposes only**. It does not constitute a medical diagnosis. All outputs must be reviewed and confirmed by a licensed OB-GYN, reproductive endocrinologist, or qualified healthcare professional. The ML models were trained on a specific clinical dataset and may not generalize to all patient populations.
