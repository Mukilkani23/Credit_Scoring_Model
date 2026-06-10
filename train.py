# ============================================================
# CREDIT SCORING MODEL — COMPLETE TRAINING SCRIPT
# Task 1 | CodeAlpha ML Internship
# Author: Mukilkani R P
# Dataset: Credit Risk Dataset (Kaggle: laotse/credit-risk-dataset)
# ============================================================
#
# HOW TO RUN:
#   Colab : Upload this file + CSV → run top to bottom
#   Local : python train.py  (after pip install -r requirements.txt)
#
# ============================================================

# ── COLAB ONLY ── Uncomment the 3 lines below and run first ──
# !pip install xgboost scikit-learn pandas matplotlib seaborn joblib
# from google.colab import files
# uploaded = files.upload()   ← upload credit_risk_dataset.csv here

# ─────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully")

# ─────────────────────────────────────────────────────────────
# SECTION 2: LOAD DATASET
# ─────────────────────────────────────────────────────────────
# 32,581 records | 12 raw features | Binary target: loan_status
# 0 = No Default (repaid)   1 = Default (failed to repay)

DATASET_PATH = 'credit_risk_dataset.csv'

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at '{DATASET_PATH}'\n"
        "Download from: https://www.kaggle.com/datasets/laotse/credit-risk-dataset\n"
        "Place the CSV in the same folder as this script, then re-run."
    )

df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# SECTION 3: EDA — EXPLORATORY DATA ANALYSIS
# Goal: Understand the data before touching it.
# Never skip EDA. It reveals class imbalance, outliers,
# missing values, and which features actually matter.
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("SECTION 3: EXPLORATORY DATA ANALYSIS")
print("="*55)

print("\n── Column Types & Non-Null Counts ──")
print(df.info())

print("\n── First 5 Rows ──")
print(df.head().to_string())

print("\n── Target Distribution ──")
vc = df['loan_status'].value_counts()
print(vc)
print(f"\nDefault Rate: {df['loan_status'].mean() * 100:.2f}%")
# ~22% default → mild class imbalance. We handle it with class_weight / scale_pos_weight

print("\n── Missing Values ──")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n── Statistical Summary ──")
print(df.describe().round(2).to_string())

# Create plots directory
os.makedirs('plots', exist_ok=True)

# ── Plot 1: Class Balance ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

vc.plot(kind='bar', ax=axes[0],
        color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
axes[0].set_title('Loan Default Count (0=No Default, 1=Default)', fontsize=12)
axes[0].set_xticklabels(['No Default', 'Default'], rotation=0)
axes[0].set_ylabel('Count')
for bar, count in zip(axes[0].patches, vc.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 200,
                 f'{count:,}', ha='center', fontsize=10)

vc.plot(kind='pie', ax=axes[1],
        labels=['No Default', 'Default'],
        autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
        startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Balance', fontsize=12)
axes[1].set_ylabel('')

plt.suptitle('Target Variable Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/01_class_balance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 1 saved: plots/01_class_balance.png")

# ── Plot 2: Correlation Heatmap ───────────────────────────────
plt.figure(figsize=(11, 7))
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'size': 9})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 2 saved: plots/02_correlation_heatmap.png")

# ── Plot 3: Feature Distributions ────────────────────────────
num_feats = ['person_age', 'person_income', 'person_emp_length',
             'loan_amnt', 'loan_int_rate', 'loan_percent_income']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(num_feats):
    data = df[feat].clip(upper=df[feat].quantile(0.99))
    axes[i].hist(data, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
    axes[i].set_title(feat.replace('_', ' ').title(), fontsize=11)
    axes[i].set_xlabel('')
plt.suptitle('Numerical Feature Distributions (clipped at 99th pct)', fontsize=13)
plt.tight_layout()
plt.savefig('plots/03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 3 saved: plots/03_feature_distributions.png")

# ── Plot 4: Default Rate by Loan Purpose ─────────────────────
intent_default = (df.groupby('loan_intent')['loan_status']
                    .mean()
                    .sort_values(ascending=False))
avg_default = df['loan_status'].mean()

plt.figure(figsize=(9, 5))
bars = plt.bar(intent_default.index, intent_default.values,
               color='#e67e22', edgecolor='black', alpha=0.85)
plt.axhline(avg_default, color='red', linestyle='--', linewidth=1.5,
            label=f'Dataset Average ({avg_default:.2%})')
plt.title('Default Rate by Loan Purpose', fontsize=13, fontweight='bold')
plt.ylabel('Default Rate')
plt.xticks(rotation=20, ha='right')
plt.legend()
for bar, val in zip(bars, intent_default.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.002,
             f'{val:.1%}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('plots/04_default_by_purpose.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 4 saved: plots/04_default_by_purpose.png")

print("\n✅ EDA complete — 4 plots saved to plots/")

# ─────────────────────────────────────────────────────────────
# SECTION 4: PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("SECTION 4: PREPROCESSING & FEATURE ENGINEERING")
print("="*55)

# ── 4.1 Handle Missing Values ────────────────────────────────
# Only two columns have NaN: person_emp_length and loan_int_rate
# Strategy: fill with median (robust to outliers, no bias introduced)
print(f"\nMissing values before: {df.isnull().sum().sum()}")
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
print(f"Missing values after:  {df.isnull().sum().sum()}")

# ── 4.2 Remove Extreme Outliers ──────────────────────────────
# Unrealistic records that would confuse the model
rows_before = len(df)
df = df[(df['person_age'] >= 18) & (df['person_age'] <= 80)]
df = df[df['person_emp_length'] <= 50]
print(f"\nOutlier rows removed: {rows_before - len(df)}")
print(f"Remaining rows: {len(df):,}")

# ── 4.3 Feature Engineering ──────────────────────────────────
# WHY: Raw columns often miss important relationships.
# We create new features that express these relationships directly.

# Feature 1: Debt-to-Income Ratio
# Captures how much of income is already committed to this loan.
# Higher ratio → borrower is over-stretched → higher default risk.
df['debt_to_income'] = df['loan_amnt'] / (df['person_income'] + 1)

# Feature 2: Young Borrower Flag
# Younger borrowers tend to have shorter credit histories → slightly riskier.
df['is_young'] = (df['person_age'] < 30).astype(int)

# Feature 3: High Interest Rate Flag
# Loans above 15% APR are typically offered to higher-risk borrowers.
df['high_interest'] = (df['loan_int_rate'] > 15).astype(int)

print("\n✅ 3 new features created:")
print("   debt_to_income  = loan_amnt / person_income")
print("   is_young        = 1 if age < 30, else 0")
print("   high_interest   = 1 if interest_rate > 15%, else 0")

# ── 4.4 Encode Categorical Columns ───────────────────────────
# ML models need numbers, not strings.
# LabelEncoder: maps each unique string → integer
# We save each encoder so app.py can apply the same mapping at inference.
categorical_cols = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

label_encoders = {}
print("\n✅ Encoding categorical columns:")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"   {col}: {mapping}")

# ── 4.5 Define Feature Matrix & Target ───────────────────────
FEATURE_COLS = [
    'person_age', 'person_income', 'person_home_ownership',
    'person_emp_length', 'loan_intent', 'loan_grade',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_default_on_file', 'cb_person_cred_hist_length',
    'debt_to_income', 'is_young', 'high_interest'
]
# 14 features total: 11 original + 3 engineered

X = df[FEATURE_COLS]
y = df['loan_status']

print(f"\nFeature matrix X: {X.shape}")
print(f"Target vector y:  {y.shape}")
print(f"Class counts: {dict(y.value_counts())}")

# ── 4.6 Train/Test Split ──────────────────────────────────────
# test_size=0.2 → 80% train, 20% test
# stratify=y    → both splits preserve the same class ratio (important for imbalanced data)
# random_state  → reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples")

# ── 4.7 Class Imbalance Weight ───────────────────────────────
# scale_pos_weight tells XGBoost: "there are X times more
# negatives than positives — pay more attention to positives"
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nXGBoost scale_pos_weight: {scale_pos_weight:.2f}")
print("(Tells the model to weight the minority class proportionally)")

# ─────────────────────────────────────────────────────────────
# SECTION 5: TRAIN MODELS USING sklearn PIPELINE
#
# WHY Pipeline?
# A Pipeline bundles preprocessing + model together into one object.
# When you call pipeline.fit(X_train), it handles everything internally.
# When you call pipeline.predict(X_new) in app.py, it applies the same
# preprocessing automatically — no chance of a train/inference mismatch.
# This eliminates the exact bug we fixed in Task 4.
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("SECTION 5: TRAINING MODELS")
print("="*55)

models = {

    # Logistic Regression: linear model, needs StandardScaler
    # class_weight='balanced' auto-adjusts weights for imbalanced classes
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=1000, C=1.0,
            random_state=42, class_weight='balanced'
        ))
    ]),

    # Decision Tree: tree-based, no scaling needed
    # max_depth limits tree depth to prevent overfitting
    'Decision Tree': Pipeline([
        ('clf', DecisionTreeClassifier(
            max_depth=8, min_samples_split=20,
            random_state=42, class_weight='balanced'
        ))
    ]),

    # Random Forest: ensemble of many trees, strong generalization
    # n_estimators=100 → 100 trees vote together
    'Random Forest': Pipeline([
        ('clf', RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=20,
            random_state=42, class_weight='balanced', n_jobs=-1
        ))
    ]),

    # XGBoost: gradient boosting, usually best performer on tabular data
    # scale_pos_weight → handles class imbalance
    # subsample + colsample_bytree → prevent overfitting
    'XGBoost': Pipeline([
        ('clf', XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', random_state=42
        ))
    ])
}

results = {}

for name, pipeline_model in models.items():
    print(f"\n[{name}] Training...", end=" ", flush=True)
    pipeline_model.fit(X_train, y_train)

    y_pred = pipeline_model.predict(X_test)
    y_prob = pipeline_model.predict_proba(X_test)[:, 1]

    results[name] = {
        'pipeline': pipeline_model,
        'y_pred':   y_pred,
        'y_prob':   y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, y_prob)
    }
    r = results[name]
    print(f"Done  |  F1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}")

print("\n✅ All 4 models trained!")

# ─────────────────────────────────────────────────────────────
# SECTION 6: EVALUATION
#
# Primary metric: ROC-AUC
# WHY: Accuracy is misleading on imbalanced data.
#      ROC-AUC measures the model's ability to distinguish
#      between classes across ALL thresholds — much more honest.
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("SECTION 6: MODEL EVALUATION")
print("="*55)

header = f"\n{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
print(header)
print("─" * 72)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")

# Best model selected by ROC-AUC (the most robust metric for this problem)
best_name = max(results, key=lambda k: results[k]['roc_auc'])
best = results[best_name]
print(f"\n🏆 Best Model: {best_name}  (ROC-AUC = {best['roc_auc']:.4f})")

# ── Confusion Matrix for Best Model ──────────────────────────
cm = confusion_matrix(y_test, best['y_pred'])
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=['No Default', 'Default']).plot(
    ax=ax, cmap='Blues', colorbar=False
)
ax.set_title(
    f'Confusion Matrix — {best_name}\n'
    f'TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}',
    fontsize=11
)
plt.tight_layout()
plt.savefig('plots/05_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Plot 5 saved: plots/05_confusion_matrix.png")
print(f"\nConfusion Matrix breakdown:")
print(f"  True Negatives  (correct no-default): {tn:,}")
print(f"  False Positives (flagged but fine):    {fp:,}")
print(f"  False Negatives (missed defaults):     {fn:,}  ← costly in real banking")
print(f"  True Positives  (caught defaults):     {tp:,}")

# ── ROC Curves for All Models ─────────────────────────────────
plt.figure(figsize=(9, 6))
palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for (name, r), color in zip(results.items(), palette):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    plt.plot(fpr, tpr, color=color, linewidth=2.5,
             label=f"{name}  (AUC = {r['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5,
         label='Random Baseline (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/06_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 6 saved: plots/06_roc_curves.png")

# ── Feature Importance ────────────────────────────────────────
best_clf = best['pipeline'].named_steps['clf']

if hasattr(best_clf, 'feature_importances_'):
    feat_imp = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': best_clf.feature_importances_
    }).sort_values('Importance', ascending=False)

elif hasattr(best_clf, 'coef_'):
    # Logistic Regression uses absolute coefficient values
    feat_imp = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': np.abs(best_clf.coef_[0])
    }).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x='Importance', y='Feature',
            palette='viridis_r', edgecolor='black')
plt.title(f'Feature Importance — {best_name}', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 7 saved: plots/07_feature_importance.png")

print("\n── Top 8 Features ──")
print(feat_imp.head(8).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SECTION 7: SAVE MODEL ARTIFACTS
#
# We save the entire Pipeline (not just the model).
# This means app.py gets the preprocessing bundled with it —
# same pattern as training, zero chance of mismatch.
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("SECTION 7: SAVING ARTIFACTS")
print("="*55)

os.makedirs('model', exist_ok=True)

# 1. The full pipeline (preprocessing + model in one object)
joblib.dump(best['pipeline'], 'model/credit_model_pipeline.pkl')
print("✅ model/credit_model_pipeline.pkl  ← load this in app.py")

# 2. Label encoders (so app.py can encode user-selected categories)
joblib.dump(label_encoders, 'model/label_encoders.pkl')
print("✅ model/label_encoders.pkl          ← categorical encoding")

# 3. Feature column list (ensures correct column order in app.py)
joblib.dump(FEATURE_COLS, 'model/feature_cols.pkl')
print("✅ model/feature_cols.pkl            ← feature order")

# 4. Text file with model performance (displayed in Streamlit)
with open('model/model_info.txt', 'w') as f:
    f.write(f"best_model={best_name}\n")
    f.write(f"roc_auc={best['roc_auc']:.4f}\n")
    f.write(f"f1={best['f1']:.4f}\n")
    f.write(f"accuracy={best['accuracy']:.4f}\n")
    f.write(f"precision={best['precision']:.4f}\n")
    f.write(f"recall={best['recall']:.4f}\n")
print("✅ model/model_info.txt              ← performance metrics")

# ── COLAB ONLY: Download your model files ────────────────────
# Uncomment these lines in Colab after training finishes:
#
# from google.colab import files
# files.download('model/credit_model_pipeline.pkl')
# files.download('model/label_encoders.pkl')
# files.download('model/feature_cols.pkl')
# files.download('model/model_info.txt')

print("\n" + "="*55)
print("🎉 TRAINING COMPLETE!")
print("="*55)
print(f"  Best Model : {best_name}")
print(f"  Accuracy   : {best['accuracy']:.4f}")
print(f"  Precision  : {best['precision']:.4f}")
print(f"  Recall     : {best['recall']:.4f}")
print(f"  F1-Score   : {best['f1']:.4f}")
print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
print("\nNext step → run:  streamlit run app.py")
