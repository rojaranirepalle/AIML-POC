import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Home Loan Data Analysis — Deep Learning Default Prediction

**Domain:** Finance
**Objective:** Build a deep learning model to predict whether a loan applicant will default, using the Home Credit-style application dataset.

The dataset is highly imbalanced (far more repaid loans than defaults) and has many features with missing values, which makes this a realistic, challenging classification problem.

**Steps covered in this notebook:**
1. Load the dataset
2. Check for null values
3. Percentage of default vs. payer (`TARGET`)
4. Balance the dataset (SMOTE)
5. Plot the balanced/imbalanced data
6. Encode columns required for the model
7. Train a deep learning model and calculate **sensitivity (recall)**
8. Calculate the **AUC-ROC**

> Note on ordering: encoding/cleaning (Step 6) is performed right after the null/target checks and *before* balancing (Step 4), because SMOTE needs a fully numeric matrix to generate synthetic samples. The step numbers in headers below match the assignment brief; the execution order is what a real pipeline requires.
""")

code("""import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    recall_score, precision_score, f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

sns.set_style("whitegrid")
pd.set_option("display.max_columns", 50)

DATA_PATH = "../CEP_1_Datasets/loan_data (1).csv"
""")

md("## Step 1: Load the dataset")
code("""df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df.head()
""")

md("## Step 2: Check for null values")
code("""null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
null_summary = pd.DataFrame({"missing_count": null_counts, "missing_pct": null_pct})
null_summary = null_summary[null_summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)

print(f"Columns with missing values: {len(null_summary)} out of {df.shape[1]}")
print(f"Total missing cells: {int(null_counts.sum())}")
null_summary.head(15)
""")

md("## Step 3: Percentage of default vs. payer (`TARGET`)")
code("""target_counts = df["TARGET"].value_counts()
target_pct = (df["TARGET"].value_counts(normalize=True) * 100).round(2)

print(f"Payers   (TARGET=0): {target_counts.get(0,0):>7,}  ({target_pct.get(0,0)}%)")
print(f"Defaults (TARGET=1): {target_counts.get(1,0):>7,}  ({target_pct.get(1,0)}%)")

plt.figure(figsize=(6,5))
ax = sns.countplot(x=df["TARGET"], hue=df["TARGET"], palette=["#4C72B0","#DD8452"], legend=False)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{int(h)}\\n({h/len(df)*100:.1f}%)", (p.get_x()+p.get_width()/2, h), ha="center", va="bottom")
plt.xticks([0,1], ["Payer (0)","Default (1)"])
plt.title("TARGET distribution — Original (Imbalanced) Data")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("01_target_distribution_imbalanced.png", dpi=140)
plt.show()
""")

md("""This confirms the dataset is **highly imbalanced**: roughly 92% of applicants repaid their loan (`TARGET=0`) and only about 8% defaulted (`TARGET=1`). A model trained naively on this data would be biased toward predicting "repaid" for everyone, which is exactly the failure mode we need to correct for.""")

md("""## Step 6: Encode the columns required for the model

Before we can balance the data or feed it to a neural network, every column must be numeric. This cell:
- drops columns that are missing more than 50% of their values (too sparse to be useful),
- imputes remaining missing values (median for numeric columns, mode for categorical columns),
- label-encodes categorical (text) columns into integers.""")

code("""y = df["TARGET"].copy()
X = df.drop(columns=["TARGET", "SK_ID_CURR"])

high_missing_cols = null_summary[null_summary["missing_pct"] > 50].index.tolist()
high_missing_cols = [c for c in high_missing_cols if c in X.columns]
X = X.drop(columns=high_missing_cols)
print(f"Dropped {len(high_missing_cols)} columns with >50% missing values")

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
print(f"Categorical columns to encode ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns: {len(num_cols)}")

for c in num_cols:
    X[c] = X[c].fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "Unknown")

label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    label_encoders[c] = le

print("Final feature matrix shape:", X.shape)
X.head()
""")

md("""Next we split into train/test **before** balancing, so the test set keeps the real-world (imbalanced) class distribution — that's the distribution the model will actually face in production, and the one we should evaluate against. Features are then standardized (zero mean, unit variance), which helps neural network training converge.""")

code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)
""")

md("## Step 4: Balance the dataset (SMOTE)")
code("""smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", pd.Series(y_train).value_counts().to_dict())
print("After  SMOTE:", pd.Series(y_train_bal).value_counts().to_dict())
""")

md("""SMOTE (**S**ynthetic **M**inority **O**versampling **TE**chnique) generates new, synthetic examples of the minority class (defaults) by interpolating between existing minority-class points and their nearest neighbours, rather than simply duplicating rows. It is applied **only to the training set** — the test set is left untouched so evaluation reflects real-world conditions.""")

md("## Step 5: Plot the balanced vs. imbalanced data")
code("""fig, axes = plt.subplots(1, 2, figsize=(11, 5))

before_counts = pd.Series(y_train).value_counts().sort_index()
sns.barplot(x=before_counts.index, y=before_counts.values, hue=before_counts.index,
            palette=["#4C72B0", "#DD8452"], legend=False, ax=axes[0])
axes[0].set_title("Training Data — Before Balancing")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Payer (0)", "Default (1)"])
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")

after_counts = pd.Series(y_train_bal).value_counts().sort_index()
sns.barplot(x=after_counts.index, y=after_counts.values, hue=after_counts.index,
            palette=["#4C72B0", "#DD8452"], legend=False, ax=axes[1])
axes[1].set_title("Training Data — After SMOTE Balancing")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Payer (0)", "Default (1)"])
axes[1].set_ylabel("Count")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")

plt.tight_layout()
plt.savefig("02_before_after_balancing.png", dpi=140)
plt.show()
""")

md("## Step 7 & 8: Deep Learning Model — Sensitivity and AUC-ROC\n\nA feed-forward neural network (fully-connected / MLP) with batch normalization and dropout for regularization.")

code("""# Explicit, stratified train/validation split. (Keras' validation_split takes the LAST
# fraction of the array *before* shuffling — since SMOTE appends synthetic minority
# rows at the end, that naive split would put almost only class-1 rows into validation.)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_bal, y_train_bal, test_size=0.1, random_state=42, stratify=y_train_bal
)
X_tr, y_tr = sk_shuffle(X_tr, y_tr, random_state=42)

n_features = X_train_bal.shape[1]
model = Sequential([
    Dense(128, activation="relu", input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
             tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.AUC(name="auc")],
)
model.summary()
""")

code("""early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True)

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=2048,
    callbacks=[early_stop],
    verbose=2,
)
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["loss"], label="train_loss")
axes[0].plot(history.history["val_loss"], label="val_loss")
axes[0].set_title("Loss over epochs"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(history.history["auc"], label="train_auc")
axes[1].plot(history.history["val_auc"], label="val_auc")
axes[1].set_title("AUC over epochs"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout()
plt.savefig("03_training_curves.png", dpi=140)
plt.show()
""")

md("""### Evaluating on the untouched, real-world-imbalanced test set

We predict probabilities on the test set (which still has the true ~92%/8% class split) and compute:
- **AUC-ROC** — overall ability of the model to rank defaulters above payers, independent of any threshold.
- **Sensitivity (Recall)** — of all the applicants who actually defaulted, what fraction did the model correctly flag? This is the single most important metric for a lender: missing a defaulter (a false negative) is usually far costlier than a false alarm.

Because the model was trained on SMOTE-balanced (50/50) data but tested on the real imbalanced distribution, the default 0.5 probability threshold is not necessarily optimal. We also compute the **ROC-optimal threshold** (Youden's J statistic: `argmax(TPR − FPR)`), which better balances sensitivity and specificity for this use case.""")

code("""y_pred_prob = model.predict(X_test_scaled, batch_size=4096).ravel()
auc_score = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

def metrics_at_threshold(thr):
    pred = (y_pred_prob >= thr).astype(int)
    cm_ = confusion_matrix(y_test, pred)
    tn_, fp_, fn_, tp_ = cm_.ravel()
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y_test, pred),
        "sensitivity_recall": recall_score(y_test, pred),
        "specificity": tn_ / (tn_ + fp_),
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred),
        "confusion_matrix": {"tn": int(tn_), "fp": int(fp_), "fn": int(fn_), "tp": int(tp_)},
    }

default_metrics = metrics_at_threshold(0.5)
j_scores = tpr - fpr
best_idx = int(np.argmax(j_scores))
best_threshold = float(thresholds[best_idx])
optimal_metrics = metrics_at_threshold(best_threshold)

print("AUC-ROC:", round(auc_score, 4))
print()
print("== @ default threshold 0.5 ==")
for k, v in default_metrics.items():
    print(f"  {k}: {v}")
print()
print(f"== @ ROC-optimal threshold {best_threshold:.4f} (Youden's J) ==")
for k, v in optimal_metrics.items():
    print(f"  {k}: {v}")
""")

code("""print("Classification report @ optimal threshold:\\n")
y_pred_opt = (y_pred_prob >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_opt, target_names=["Payer", "Default"]))
""")

code("""cm = confusion_matrix(y_test, y_pred_opt)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Payer", "Default"], yticklabels=["Payer", "Default"])
plt.title(f"Confusion Matrix (Test Set, threshold={best_threshold:.3f})")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("04_confusion_matrix.png", dpi=140)
plt.show()
""")

code("""plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color="#4C72B0")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.scatter([fpr[best_idx]], [tpr[best_idx]], color="#DD8452", zorder=5,
            label=f"Optimal threshold={best_threshold:.3f}\\n(sensitivity={tpr[best_idx]:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("05_roc_curve.png", dpi=140)
plt.show()
""")

md("""## Summary

| Metric | @ threshold 0.5 | @ ROC-optimal threshold |
|---|---|---|
| Sensitivity (Recall) | see output above | see output above |
| Specificity | see output above | see output above |
| AUC-ROC | *(threshold-independent)* {:.3f} | |

**Key takeaways:**
- The raw dataset is highly imbalanced (~92% payers vs ~8% defaults).
- SMOTE was used to balance the *training* data only, so the model sees an equal number of examples from both classes during learning, while the test set still reflects reality.
- AUC-ROC measures the model's overall discriminative power independent of a decision threshold.
- Sensitivity (recall) — the fraction of actual defaulters correctly identified — is the more business-relevant metric for a lender, since failing to catch a defaulter is typically far costlier than a false alarm on a good applicant. Tuning the classification threshold (e.g. via Youden's J) is a simple way to trade some specificity for meaningfully higher sensitivity.
- Further improvements could include: hyperparameter tuning, deeper architectures, feature engineering from the auxiliary Home Credit tables (bureau, previous applications, etc.), or class-weighted loss as an alternative/complement to SMOTE.
""")

nb['cells'] = cells
os.makedirs(os.path.expanduser("~/mnt/Project 1 - Home Loan Data Analysis/outputs"), exist_ok=True)
out_path = os.path.expanduser("~/mnt/Project 1 - Home Loan Data Analysis/outputs/Home_Loan_Default_Analysis.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print("Notebook written to:", out_path)
