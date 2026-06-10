# 💳 Credit Scoring Model

**CodeAlpha ML Internship — Task 1**  
Predicts loan default risk using an applicant's financial and personal profile.

---

## 📌 Problem Statement

Given a loan applicant's financial data (income, loan amount, credit history, etc.),
predict whether they will **default on the loan** (1) or **repay it** (0).

This is a binary classification problem with mild class imbalance (~22% default rate).

---

## 📊 Dataset

- **Source:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Records:** 32,581
- **Features:** 12 raw → 14 after feature engineering
- **Target:** `loan_status` (0 = No Default, 1 = Default)

| Feature | Type | Description |
|---------|------|-------------|
| person_age | int | Applicant age |
| person_income | float | Annual income |
| person_home_ownership | cat | RENT / OWN / MORTGAGE / OTHER |
| person_emp_length | float | Employment length in years |
| loan_intent | cat | Loan purpose |
| loan_grade | cat | Bank-assigned grade A–G |
| loan_amnt | float | Loan amount requested |
| loan_int_rate | float | Interest rate (%) |
| loan_percent_income | float | Loan as % of income |
| cb_person_default_on_file | cat | Historical default Y/N |
| cb_person_cred_hist_length | int | Years of credit history |
| loan_status | int | **Target** — 0 or 1 |

---

## ⚙️ Tech Stack

- Python 3.9+
- scikit-learn (Logistic Regression, Decision Tree, Random Forest, Pipeline)
- XGBoost
- pandas, numpy, matplotlib, seaborn
- joblib (model serialization)
- Streamlit (web app)

---

## 🗂️ Project Structure

```
credit-scoring-model/
├── train.py                     ← Full training script
├── app.py                       ← Streamlit web app
├── requirements.txt
├── .gitignore
├── model/                       ← Generated after training
│   ├── credit_model_pipeline.pkl
│   ├── label_encoders.pkl
│   ├── feature_cols.pkl
│   └── model_info.txt
└── plots/                       ← Generated after training
    ├── 01_class_balance.png
    ├── 02_correlation_heatmap.png
    ├── 03_feature_distributions.png
    ├── 04_default_by_purpose.png
    ├── 05_confusion_matrix.png
    ├── 06_roc_curves.png
    └── 07_feature_importance.png
```

---

## 🚀 How to Run

### Step 1 — Clone the repo
```bash
git clone https://github.com/Mukilkani23/CodeAlpha-Credit-Scoring-Model.git
cd CodeAlpha-Credit-Scoring-Model
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the dataset
Download `credit_risk_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  
Place it in the project root (same folder as `train.py`).

### Step 4 — Train the model
```bash
python train.py
```

### Step 5 — Launch the app
```bash
streamlit run app.py
```

---

## 📈 Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | **—** |

> *(Fill these in after running train.py)*

---

## 🌐 Live Demo

[Click here to try the app](#) ← *(add Streamlit Cloud link after deployment)*

---

## 🧠 Key Learnings

- **sklearn Pipeline** prevents train/inference mismatch — preprocessing is baked into the model
- **ROC-AUC > Accuracy** as primary metric for imbalanced classification
- **scale_pos_weight** in XGBoost and **class_weight='balanced'** in sklearn handle class imbalance
- Feature engineering (debt_to_income, is_young, high_interest) improves model signal

---

## 👤 Author

**Mukilkani R P**  
B.E. CSE 
[GitHub](https://github.com/Mukilkani23) · [LinkedIn]()
