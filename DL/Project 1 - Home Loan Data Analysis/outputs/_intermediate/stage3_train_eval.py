import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    recall_score, precision_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

BASE = os.path.expanduser("~/mnt/Project 1 - Home Loan Data Analysis")
OUT = os.path.join(BASE, "outputs")

t0 = time.time()
d1 = np.load(os.path.join(OUT, "stage1_data.npz"))
d2 = np.load(os.path.join(OUT, "stage2_balanced.npz"))
X_test, y_test = d1["X_test"], d1["y_test"]
X_train_bal, y_train_bal = d2["X_train_bal"], d2["y_train_bal"]
print(f"Loaded data in {time.time()-t0:.1f}s. Train_bal {X_train_bal.shape}, Test {X_test.shape}")

# NOTE: SMOTE appends synthetic minority samples at the end of the array.
# Keras' validation_split takes the LAST x% of samples *before* shuffling,
# which would make the validation set almost entirely one class. So we
# build an explicit, stratified train/validation split instead.
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_bal, y_train_bal, test_size=0.1, random_state=42, stratify=y_train_bal
)
X_tr, y_tr = sk_shuffle(X_tr, y_tr, random_state=42)
print(f"Train fit shape: {X_tr.shape}, Val shape: {X_val.shape}")

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

early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True)

t1 = time.time()
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=2048,
    callbacks=[early_stop],
    verbose=2,
)
print(f"Training done in {time.time()-t1:.1f}s")

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

y_pred_prob = model.predict(X_test, batch_size=4096).ravel()
auc_score = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

def metrics_at_threshold(thr):
    pred = (y_pred_prob >= thr).astype(int)
    cm_ = confusion_matrix(y_test, pred)
    tn_, fp_, fn_, tp_ = cm_.ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_test, pred)),
        "sensitivity_recall": float(recall_score(y_test, pred)),
        "specificity": float(tn_ / (tn_ + fp_)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred)),
        "confusion_matrix": {"tn": int(tn_), "fp": int(fp_), "fn": int(fn_), "tp": int(tp_)},
    }

# Default threshold (0.5)
default_metrics = metrics_at_threshold(0.5)

# Optimal threshold via Youden's J statistic (maximizes sensitivity + specificity)
j_scores = tpr - fpr
best_idx = int(np.argmax(j_scores))
best_threshold = float(thresholds[best_idx])
optimal_metrics = metrics_at_threshold(best_threshold)

print("\n=== TEST-SET METRICS @ default threshold 0.5 ===")
for k, v in default_metrics.items():
    print(f"{k}: {v}")
print(classification_report(y_test, (y_pred_prob >= 0.5).astype(int), target_names=["Payer", "Default"]))

print(f"\n=== TEST-SET METRICS @ optimal threshold {best_threshold:.4f} (Youden's J) ===")
for k, v in optimal_metrics.items():
    print(f"{k}: {v}")
print(classification_report(y_test, (y_pred_prob >= best_threshold).astype(int), target_names=["Payer", "Default"]))

print(f"\nAUC-ROC: {auc_score:.4f}")

sensitivity = optimal_metrics["sensitivity_recall"]
cm = np.array([[optimal_metrics["confusion_matrix"]["tn"], optimal_metrics["confusion_matrix"]["fp"]],
               [optimal_metrics["confusion_matrix"]["fn"], optimal_metrics["confusion_matrix"]["tp"]]])
y_pred = (y_pred_prob >= best_threshold).astype(int)

with open(os.path.join(OUT, "final_metrics.json"), "w") as f:
    json.dump({
        "auc_roc": float(auc_score),
        "epochs_trained": len(history.history["loss"]),
        "at_default_threshold_0.5": default_metrics,
        "at_optimal_threshold_youdens_j": optimal_metrics,
    }, f, indent=2)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Payer", "Default"], yticklabels=["Payer", "Default"])
plt.title(f"Confusion Matrix (Test Set, threshold={best_threshold:.3f})")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_confusion_matrix.png"), dpi=140)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color="#4C72B0")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.scatter([fpr[best_idx]], [tpr[best_idx]], color="#DD8452", zorder=5,
            label=f"Optimal threshold={best_threshold:.3f}\n(sensitivity={tpr[best_idx]:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_roc_curve.png"), dpi=140)
plt.close()

model.save(os.path.join(OUT, "home_loan_default_model.keras"))
print(f"\nTotal stage3 time: {time.time()-t0:.1f}s")
print("STAGE3_OK")
