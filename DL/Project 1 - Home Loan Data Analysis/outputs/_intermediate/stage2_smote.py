import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

BASE = os.path.expanduser("~/mnt/Project 1 - Home Loan Data Analysis")
OUT = os.path.join(BASE, "outputs")

t0 = time.time()
data = np.load(os.path.join(OUT, "stage1_data.npz"))
X_train, y_train = data["X_train"], data["y_train"]
print(f"Loaded stage1 data in {time.time()-t0:.1f}s. X_train {X_train.shape}")

print("Running SMOTE...")
t1 = time.time()
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"SMOTE done in {time.time()-t1:.1f}s")
print("Before:", pd.Series(y_train).value_counts().to_dict())
print("After :", pd.Series(y_train_bal).value_counts().to_dict())

np.savez_compressed(os.path.join(OUT, "stage2_balanced.npz"),
                     X_train_bal=X_train_bal.astype(np.float32),
                     y_train_bal=y_train_bal.astype(np.int8))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
before_counts = pd.Series(y_train).value_counts().sort_index()
sns.barplot(x=before_counts.index, y=before_counts.values, hue=before_counts.index,
            palette=["#4C72B0", "#DD8452"], legend=False, ax=axes[0])
axes[0].set_title("Training Data - Before Balancing")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Payer (0)", "Default (1)"])
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")

after_counts = pd.Series(y_train_bal).value_counts().sort_index()
sns.barplot(x=after_counts.index, y=after_counts.values, hue=after_counts.index,
            palette=["#4C72B0", "#DD8452"], legend=False, ax=axes[1])
axes[1].set_title("Training Data - After SMOTE Balancing")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Payer (0)", "Default (1)"])
axes[1].set_ylabel("Count")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_before_after_balancing.png"), dpi=140)
plt.close()

print(f"Stage 2 total time: {time.time()-t0:.1f}s")
print("STAGE2_OK")
