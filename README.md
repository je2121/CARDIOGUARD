# 🫀 CardioGuard AI: Cardiovascular Risk Prediction & Explainable AI

CardioGuard AI is an end-to-end Machine Learning web application designed to predict cardiovascular disease (CVD) event risks. Built using **Streamlit** and an **XGBoost Classifier**, the application handles custom feature engineering to assess patient vitals and lifestyle choices. 

To eliminate standard "black-box" limitations in healthcare AI, this project integrates **SHAP (SHapley Additive exPlanations)** to break down exactly *how* and *why* a specific prediction was made, converting complex tree-log-odds metrics directly into readable risk probability changes.

---

## ✨ Features

* **Risk Evaluation:** Instantaneously calculates cardiovascular risk percentage and categorizes patients into Low, Moderate, or High-risk brackets.
* **Explainable AI Dashboard:** Features an interactive SHAP layout breaking down feature impact (Red bars accelerate risk, Blue bars lower risk/act as protective factors).
* **Interactive Data Visualizations:** Includes fully dynamic Plotly gauge charts and horizontal contribution charts for crystal-clear medical interpretations.
* **Advanced Feature Engineering:** Calculates derived physiological interaction indicators on the fly, including Pulse Pressure ($Systolic - Diastolic$) and Age-to-BMI ratios.

---

## 🛠️ Technology Stack

* **User Interface Framework:** Streamlit
* **Core ML Predictor Engine:** XGBoost (Extreme Gradient Boosting)
* **Model Explainability Layer:** SHAP Core Infrastructure
* **Data Processing & Graphics:** Pandas, NumPy, Plotly
* **Model Serialization:** Joblib

---

## 📂 Repository Structure

The workspace directory is configured as follows to ensure flawless web server mapping:

```text
CardioGuard-AI/
│
├── app.py                   # Main Streamlit web application script
├── requirements.txt         # Production server dependencies list
├── .gitignore               # System cache and environment exclusion filters
├── README.md                # Comprehensive project documentation
├── notebook.ipynb           # Jupyter notebook detailing model training and evaluation
│
└── models/
    └── xgb_cvd_model.pkl    # Serialized binary snapshot of the trained XGBoost model
