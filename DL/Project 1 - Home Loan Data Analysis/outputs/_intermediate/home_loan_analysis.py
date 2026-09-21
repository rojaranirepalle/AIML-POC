"""
Home Loan Data Analysis - Deep Learning default prediction
Steps:
1. Load dataset
2. Check nulls
3. Print % default vs payer for TARGET
4. Balance the dataset (SMOTE)
5. Plot balanced/imbalanced data
6. Encode columns
7. Build DL model, calculate sensitivity (recall)
8. Calculate AUC-ROC
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    recall_score, precision_score, f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

BASE = os.path.expanduser("~/mnt/Project 1 - Home Loan Data Analysis")
DATA_PATH = os.path.join(BASE, "CEP_1_Datasets", "loan_data (1).csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

log = []
def logp(msg):
    print(msg, flush=True)
    log.append(str(msg))

t0 = time.time()

# ---------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------
logp("STEP 1: Loading dataset...")
df = pd.read_csv(DATA_PATH)
logp(f"Shape: {df.shape}")
logp(f"Loaded in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------
# 2. Check for null values
# ---------------------------------------------------------------
logp("\nSTEP 2: Checking null values...")
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
null_summary = pd.DataFrame({"missing_count": null_counts, "missing_pct": null_pct})
null_summary = null_summary[null_summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)
null_summary.to_csv(os.path.join(OUT, "null_value_summary.csv"))
logp(f"Total columns with missing values: {len(null_summary)} out of {df.shape[1]}")
logp(f"Total missing cells: {int(null_counts.sum())}")
logp(null_summary.head(15).to_string())

# ---------------------------------------------------------------
# 3. Percentage of default vs payer (TARGET)
# ---------------------------------------------------------------
logp("\nSTEP 3: TARGET distribution (default vs payer)...")
target_counts = df["TARGET"].value_counts()
target_pct = (df["TARGET"].value_counts(normalize=True) * 100).round(2)
logp(f"Payers (TARGET=0): {target_counts.get(0,0)} ({target_pct.get(0,0)}%)")
logp(f"Defaulters (TARGET=1): {target_counts.get(1,0)} ({target_pct.get(1,0)}%)")

with open(os.path.join(OUT, "target_distribution.json"), "w") as f:
    json.dump({
        "payer_count": int(target_counts.get(0, 0)),
        "payer_pct": float(target_pct.get(0, 0)),
        "default_count": int(target_counts.get(1, 0)),
        "default_pct": float(target_pct.get(1, 0)),
    }, f, indent=2)

# Plot ORIGINAL (imbalanced) distribution
plt.figure(figsize=(6, 5))
ax = sns.countplot(x=df["TARGET"], hue=df["TARGET"], palette=["#4C72B0", "#DD8452"], legend=False)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{int(h)}\n({h/len(df)*100:.1f}%)", (p.get_x() + p.get_width()/2, h),
                ha="center", va="bottom", fontsize=10)
plt.xticks([0, 1], ["Payer (0)", "Default (1)"])
plt.title("TARGET distribution - Original (Imbalanced) Data")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_target_distribution_imbalanced.png"), dpi=140)
plt.close()

# ---------------------------------------------------------------
# 6. Preprocessing / Encoding (done before balancing so SMOTE gets numeric data)
# ---------------------------------------------------------------
logp("\nSTEP 6 (prep): Cleaning + Encoding columns...")

y = df["TARGET"].copy()
X = df.drop(columns=["TARGET", "SK_ID_CURR"])

# Drop columns with too much missing data (>50%) to keep the model tractable
high_missing_cols = null_summary[null_summary["missing_pct"] > 50].index.tolist()
high_missing_cols = [c for c in high_missing_cols if c in X.columns]
X = X.drop(columns=high_missing_cols)
logp(f"Dropped {len(high_missing_cols)} columns with >50% missing values")

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
logp(f"Categorical columns to encode: {len(cat_cols)}")
logp(f"Numeric columns: {len(num_cols)}")

# Impute
for c in num_cols:
    X[c] = X[c].fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "Unknown")

# Label-encode categorical columns
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    encoders[c] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

logp(f"Final feature matrix shape: {X.shape}")

# Train/test split (stratified) BEFORE balancing -> balance only the training set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logp(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Balance the dataset (SMOTE on training data only)
# ---------------------------------------------------------------
logp("\nSTEP 4: Balancing dataset with SMOTE...")
t1 = time.time()
smote = SMOTE(random_state=42, n_jobs=-1)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
logp(f"SMOTE completed in {time.time()-t1:.1f}s")
logp(f"Before SMOTE: {y_train.value_counts().to_dict()}")
logp(f"After SMOTE:  {pd.Series(y_train_bal).value_counts().to_dict()}")

# ---------------------------------------------------------------
# 5. Plot balanced data
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
before_counts = y_train.value_counts().sort_index()
sns.barplot(x=before_counts.index, y=before_counts.values, hue=before_counts.index,
            palette=["#4C72B0", "#DD8452"], legend=False, ax=axes[0])
axes[0].set_title("Training Data - Before Balancing")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Payer (0)", "Default (1)"])
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                      ha="center", va="bottom")

after_counts = pd.Series(y_train_bal).value_counts().sort_index()
sns.barplot(x=after_counts.index, y=after_counts.values, hue=after_counts.index,
            palette=["#4C72B0", "#DD8452"], legend=False, ax=axes[1])
axes[1].set_title("Training Data - After SMOTE Balancing")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Payer (0)", "Default (1)"])
axes[1].set_ylabel("Count")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                      ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_before_after_balancing.png"), dpi=140)
plt.close()

# ---------------------------------------------------------------
# 7 & 8. Deep learning model -> Sensitivity (Recall) & AUC-ROC
# ---------------------------------------------------------------
logp("\nSTEP 7/8: Building and training the deep learning model...")

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
    metrics=[
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),        # sensitivity
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True)

history = model.fit(
    X_train_bal, y_train_bal,
    validation_split=0.1,
    epochs=25,
    batch_size=1024,
    callbacks=[early_stop],
    verbose=2,
)

t2 = time.time()
logp(f"Training completed in {t2 - t1:.1f}s total")

# Save training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["loss"], label="train_loss")
axes[0].plot(history.history["val_loss"], label="val_loss")
axes[0].set_title("Loss over epochs"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(history.history["auc"], label="train_auc")
axes[1].plot(history.history["val_auc"], label="val_auc")
axes[1].set_title("AUC over epochs"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_training_curves.png"), dpi=140)
plt.close()

# Evaluate on the untouched, ORIGINAL-distribution test set
y_pred_prob = model.predict(X_test_scaled, batch_size=2048).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

sensitivity = recall_score(y_test, y_pred)           # STEP 7
auc_score = roc_auc_score(y_test, y_pred_prob)        # STEP 8
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

logp("\n=== FINAL TEST-SET METRICS (original imbalanced distribution) ===")
logp(f"Accuracy:            {accuracy:.4f}")
logp(f"Sensitivity (Recall):{sensitivity:.4f}")
logp(f"Specificity:         {specificity:.4f}")
logp(f"Precision:           {precision:.4f}")
logp(f"F1-score:            {f1:.4f}")
logp(f"AUC-ROC:             {auc_score:.4f}")
logp(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
logp(classification_report(y_test, y_pred, target_names=["Payer", "Default"]))

with open(os.path.join(OUT, "final_metrics.json"), "w") as f:
    json.dump({
        "accuracy": float(accuracy),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "auc_roc": float(auc_score),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }, f, indent=2)

# Confusion matrix plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Payer", "Default"], yticklabels=["Payer", "Default"])
plt.title("Confusion Matrix (Test Set)")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_confusion_matrix.png"), dpi=140)
plt.close()

# ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color="#4C72B0")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_roc_curve.png"), dpi=140)
plt.close()

model.save(os.path.join(OUT, "home_loan_default_model.keras"))

with open(os.path.join(OUT, "run_log.txt"), "w") as f:
    f.write("\n".join(log))

logp(f"\nALL DONE. Total time: {time.time()-t0:.1f}s")
logp("DONE_MARKER_OK")
