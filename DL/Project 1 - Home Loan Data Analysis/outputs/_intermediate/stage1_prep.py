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

BASE = os.path.expanduser("~/mnt/Project 1 - Home Loan Data Analysis")
DATA_PATH = os.path.join(BASE, "CEP_1_Datasets", "loan_data (1).csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
log = []
def logp(m):
    print(m, flush=True); log.append(str(m))

logp("STEP 1: Loading dataset...")
df = pd.read_csv(DATA_PATH)
logp(f"Shape: {df.shape}  (loaded in {time.time()-t0:.1f}s)")

logp("\nSTEP 2: Checking null values...")
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
null_summary = pd.DataFrame({"missing_count": null_counts, "missing_pct": null_pct})
null_summary = null_summary[null_summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)
null_summary.to_csv(os.path.join(OUT, "null_value_summary.csv"))
logp(f"Columns with missing values: {len(null_summary)} / {df.shape[1]}")
logp(f"Total missing cells: {int(null_counts.sum())}")

logp("\nSTEP 3: TARGET distribution (default vs payer)...")
target_counts = df["TARGET"].value_counts()
target_pct = (df["TARGET"].value_counts(normalize=True) * 100).round(2)
logp(f"Payers (TARGET=0): {target_counts.get(0,0)} ({target_pct.get(0,0)}%)")
logp(f"Defaulters (TARGET=1): {target_counts.get(1,0)} ({target_pct.get(1,0)}%)")
with open(os.path.join(OUT, "target_distribution.json"), "w") as f:
    json.dump({"payer_count": int(target_counts.get(0,0)), "payer_pct": float(target_pct.get(0,0)),
               "default_count": int(target_counts.get(1,0)), "default_pct": float(target_pct.get(1,0))}, f, indent=2)

plt.figure(figsize=(6,5))
ax = sns.countplot(x=df["TARGET"], hue=df["TARGET"], palette=["#4C72B0","#DD8452"], legend=False)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{int(h)}\n({h/len(df)*100:.1f}%)", (p.get_x()+p.get_width()/2, h), ha="center", va="bottom")
plt.xticks([0,1], ["Payer (0)","Default (1)"])
plt.title("TARGET distribution - Original (Imbalanced) Data")
plt.ylabel("Count"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_target_distribution_imbalanced.png"), dpi=140)
plt.close()

logp("\nSTEP 6 (prep): Cleaning + Encoding columns...")
y = df["TARGET"].copy()
X = df.drop(columns=["TARGET", "SK_ID_CURR"])

high_missing_cols = null_summary[null_summary["missing_pct"] > 50].index.tolist()
high_missing_cols = [c for c in high_missing_cols if c in X.columns]
X = X.drop(columns=high_missing_cols)
logp(f"Dropped {len(high_missing_cols)} columns with >50% missing values")

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
logp(f"Categorical columns to encode: {len(cat_cols)} -> {cat_cols}")
logp(f"Numeric columns: {len(num_cols)}")

for c in num_cols:
    X[c] = X[c].fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "Unknown")

encoders_info = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    encoders_info[c] = le.classes_.tolist()

with open(os.path.join(OUT, "label_encoders.json"), "w") as f:
    json.dump(encoders_info, f)

feature_names = X.columns.tolist()
with open(os.path.join(OUT, "feature_names.json"), "w") as f:
    json.dump(feature_names, f)

logp(f"Final feature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logp(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

np.savez_compressed(os.path.join(OUT, "stage1_data.npz"),
                     X_train=X_train_scaled, X_test=X_test_scaled,
                     y_train=y_train.values.astype(np.int8), y_test=y_test.values.astype(np.int8))

logp(f"\nStage 1 done in {time.time()-t0:.1f}s")
with open(os.path.join(OUT, "stage1_log.txt"), "w") as f:
    f.write("\n".join(log))
print("STAGE1_OK")
