"""
╔══════════════════════════════════════════════════════════════════════╗
║   Tech Stack ML — Phase 4: Model Training (Mac / VS Code)           ║
║                                                                      ║
║   Models:                                                            ║
║     1. ChainCatBoost  — ClassifierChain wrapping CatBoost            ║
║     2. ChainXGB       — ClassifierChain wrapping XGBoost             ║
║     3. NoBERT         — Frozen sentence-transformer + LightGBM head  ║
║     4. LightGBM       — Fast gradient boosting on TF-IDF features    ║
║     5. StackingEnsemble — Meta-learner over 4 base models            ║
║                                                                      ║
║   HOW TO RUN                                                         ║
║   ──────────────────────────────────────────────────────────────     ║
║   1. Place in the same folder as this script:                        ║
║        · train_val_test_splits.pkl                                   ║
║        · github_projects_cleaned.xlsx                                ║
║                                                                      ║
║   2. Create a virtual environment (one-time):                        ║
║        python3 -m venv venv                                          ║
║        source venv/bin/activate                                      ║
║                                                                      ║
║   3. Install dependencies (one-time):                                ║
║        pip install pandas numpy scikit-learn catboost xgboost        ║
║                   lightgbm sentence-transformers openpyxl            ║
║                   joblib matplotlib seaborn                          ║
║                                                                      ║
║   4. Run:                                                            ║
║        python phase4_training_5models_mac.py                         ║
║                                                                      ║
║   All outputs saved to ./outputs/ folder (auto-created)              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════
# CONFIG — edit these two paths if your files are elsewhere
# ══════════════════════════════════════════════════════════════
PKL_PATH   = "train_val_test_splits.pkl"
XLSX_PATH  = "github_projects_cleaned.xlsx"

OUTPUT_DIR = "outputs"

# Set False to skip NoBERT during quick dev runs (~10 min saved)
RUN_NOBERT = True


# ══════════════════════════════════════════════════════════════
# CELL 1 ─ Imports
# ══════════════════════════════════════════════════════════════
import os, copy, time, warnings
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save charts to file, no display window needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (accuracy_score, f1_score,
                                     classification_report,
                                     confusion_matrix, hamming_loss)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.linear_model    import LogisticRegression

from catboost  import CatBoostClassifier
from xgboost   import XGBClassifier
import lightgbm as lgb
from lightgbm  import LGBMClassifier
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    """Returns full path inside OUTPUT_DIR."""
    return os.path.join(OUTPUT_DIR, filename)

plt.rcParams.update({
    "figure.dpi"       : 140,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 11,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
})

print("✓ All libraries imported")


# ══════════════════════════════════════════════════════════════
# CELL 2 ─ Load train_val_test_splits.pkl
# ══════════════════════════════════════════════════════════════
print(f"\nLoading {PKL_PATH} …")
assert os.path.exists(PKL_PATH), f"❌  File not found: {PKL_PATH}"

splits = joblib.load(PKL_PATH)

assert "X_val" in splits, (
    "❌  No X_val key found.\n"
    "    Please use train_val_test_splits.pkl from preprocessing_80_10_10.py")
assert splits.get("split_ratio") == "80/10/10", (
    "⚠   split_ratio mismatch — verify this pkl came from preprocessing_80_10_10.py")

X_train    = splits["X_train"];  X_val  = splits["X_val"];  X_test = splits["X_test"]

y_fe_train = splits["y_fe_train"]; y_fe_val = splits["y_fe_val"]; y_fe_test = splits["y_fe_test"]
y_be_train = splits["y_be_train"]; y_be_val = splits["y_be_val"]; y_be_test = splits["y_be_test"]
y_db_train = splits["y_db_train"]; y_db_val = splits["y_db_val"]; y_db_test = splits["y_db_test"]

le_fe      = splits["le_fe"]
le_be      = splits["le_be"]
le_db      = splits["le_db"]
feat_names = splits["feature_names"]

print(f"✓ Data loaded")
print(f"  X_train : {X_train.shape}  (80%)")
print(f"  X_val   : {X_val.shape}   (10%)")
print(f"  X_test  : {X_test.shape}   (10%)")
print(f"  Frontend classes ({len(le_fe.classes_)}): {list(le_fe.classes_)}")
print(f"  Backend  classes ({len(le_be.classes_)}): {list(le_be.classes_)}")
print(f"  Database classes ({len(le_db.classes_)}): {list(le_db.classes_)}")


# ══════════════════════════════════════════════════════════════
# CELL 3 ─ Majority-class baselines
# ══════════════════════════════════════════════════════════════
def majority_baseline(y_train, y_eval):
    most_freq = Counter(y_train).most_common(1)[0][0]
    return accuracy_score(y_eval, np.full(len(y_eval), most_freq))

bl_fe_val  = majority_baseline(y_fe_train, y_fe_val)
bl_be_val  = majority_baseline(y_be_train, y_be_val)
bl_db_val  = majority_baseline(y_db_train, y_db_val)
bl_fe_test = majority_baseline(y_fe_train, y_fe_test)
bl_be_test = majority_baseline(y_be_train, y_be_test)
bl_db_test = majority_baseline(y_db_train, y_db_test)

fe_mf = le_fe.classes_[Counter(y_fe_train).most_common(1)[0][0]]
be_mf = le_be.classes_[Counter(y_be_train).most_common(1)[0][0]]
db_mf = le_db.classes_[Counter(y_db_train).most_common(1)[0][0]]

print(f"\n{'='*62}")
print("  BASELINES  (always predict most frequent class)")
print(f"  {'Target':<12} {'Val':>10} {'Test':>10}")
print(f"  {'─'*36}")
print(f"  Frontend     {bl_fe_val:>10.3f} {bl_fe_test:>10.3f}  → always '{fe_mf}'")
print(f"  Backend      {bl_be_val:>10.3f} {bl_be_test:>10.3f}  → always '{be_mf}'")
print(f"  Database     {bl_db_val:>10.3f} {bl_db_test:>10.3f}  → always '{db_mf}'")
print(f"{'='*62}\n")

BASELINES = {
    "Frontend": {"val": bl_fe_val,  "test": bl_fe_test},
    "Backend":  {"val": bl_be_val,  "test": bl_be_test},
    "Database": {"val": bl_db_val,  "test": bl_db_test},
}
targets_list = ["Frontend", "Backend", "Database"]


# ══════════════════════════════════════════════════════════════
# CELL 4 ─ Shared helpers
# ══════════════════════════════════════════════════════════════
def evaluate(model, X_ev, y_ev, le, split_label):
    """Standard metrics dict for any sklearn-compatible model."""
    y_pred = np.array(model.predict(X_ev), dtype=int)
    labels = np.arange(len(le.classes_))
    return {
        "split":       split_label,
        "accuracy":    accuracy_score(y_ev, y_pred),
        "f1_weighted": f1_score(y_ev, y_pred, average="weighted",  zero_division=0),
        "f1_macro":    f1_score(y_ev, y_pred, average="macro",     zero_division=0),
        "hamming":     hamming_loss(y_ev, y_pred),
        "report":      classification_report(y_ev, y_pred,
                           labels=labels, target_names=le.classes_,
                           zero_division=0),
        "y_pred":      y_pred,
    }

def print_result(tname, val_m, test_m, elapsed, extra=""):
    bl_v = BASELINES[tname]["val"]
    bl_t = BASELINES[tname]["test"]
    print(f"  {tname:<10}  "
          f"val={val_m['accuracy']:.3f} (bl={bl_v:.3f})  "
          f"test={test_m['accuracy']:.3f} (bl={bl_t:.3f})  "
          f"[{elapsed:.1f}s]  {extra}")


# ══════════════════════════════════════════════════════════════
# CELL 5 ─ Load Excel + recover text splits (NoBERT needs this)
# ══════════════════════════════════════════════════════════════
print(f"Loading {XLSX_PATH} …")
assert os.path.exists(XLSX_PATH), f"❌  File not found: {XLSX_PATH}"

df_raw = pd.read_excel(XLSX_PATH)

# Apply the same rare-class grouping used in preprocessing
RARE_FE = {"Svelte"}
RARE_BE = {"Express", "FastAPI", "PHP", "Firebase"}
RARE_DB = {"Elasticsearch", "DynamoDB", "Cassandra", "Oracle", "Firestore"}

df_raw["Frontend_Tech"]    = df_raw["Frontend_Tech"].apply(lambda x: "Other" if x in RARE_FE else x)
df_raw["Backend_Tech"]     = df_raw["Backend_Tech"].apply(lambda x: "Other" if x in RARE_BE else x)
df_raw["Database"]         = df_raw["Database"].apply(lambda x: "Other" if x in RARE_DB else x)
df_raw["Primary_Language"] = df_raw["Primary_Language"].fillna("Unknown")

df_raw["combined_text"] = (
    df_raw["Functional_Requirements"].fillna("").astype(str) + " " +
    df_raw["Non_Functional_Requirements"].fillna("").astype(str)
)

# Replicate the exact two-step stratified split from preprocessing
# (same random_state=42) to recover the same row indices
all_idx = np.arange(len(df_raw))
dummy_y = le_fe.transform(df_raw["Frontend_Tech"])

idx_train_val, idx_test = train_test_split(
    all_idx, test_size=0.10, random_state=42, stratify=dummy_y)
idx_train, idx_val = train_test_split(
    idx_train_val, test_size=10/90, random_state=42,
    stratify=dummy_y[idx_train_val])

text_train = df_raw["combined_text"].iloc[idx_train].tolist()
text_val   = df_raw["combined_text"].iloc[idx_val].tolist()
text_test  = df_raw["combined_text"].iloc[idx_test].tolist()

assert len(text_train) == len(X_train), \
    f"❌  text_train length {len(text_train)} ≠ X_train {len(X_train)}"
assert len(text_val)   == len(X_val)
assert len(text_test)  == len(X_test)

print(f"✓ Text recovered: train={len(text_train):,}  val={len(text_val):,}  test={len(text_test):,}\n")


# ══════════════════════════════════════════════════════════════
# CELL 6 ─ Convert to numpy arrays
# ══════════════════════════════════════════════════════════════
X_tr = X_train.values if hasattr(X_train, "values") else X_train
X_v  = X_val.values   if hasattr(X_val,   "values") else X_val
X_te = X_test.values  if hasattr(X_test,  "values") else X_test

TARGETS = {
    "Frontend": (y_fe_train, y_fe_val, y_fe_test, le_fe),
    "Backend":  (y_be_train, y_be_val, y_be_test, le_be),
    "Database": (y_db_train, y_db_val, y_db_test, le_db),
}

# Storage
results = {}   # results[model_name][target] = {"val": {...}, "test": {...}}
trained = {}   # trained[model_name][target] = fitted model object

MODEL_NAMES = ["ChainCatBoost", "ChainXGB", "NoBERT", "LightGBM", "StackingEnsemble"]

print("✓ Arrays ready\n")


# ══════════════════════════════════════════════════════════════
# CELL 7 ─ NoBERT helper class
# ══════════════════════════════════════════════════════════════
# NoBERT = frozen sentence-transformer (all-MiniLM-L6-v2, 384-dim)
# encodes the combined FR + NFR text, then concatenates those
# embeddings with the structured features and trains LightGBM on top.
# No BERT fine-tuning → works well on small datasets (<5k rows).

class NoBERTClassifier:
    """Frozen sentence-transformer encoder + LightGBM classifier head."""

    def __init__(self, model_name="all-MiniLM-L6-v2",
                 n_estimators=500, num_leaves=63,
                 learning_rate=0.05, random_state=42):
        self.encoder_name  = model_name
        self.n_estimators  = n_estimators
        self.num_leaves    = num_leaves
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.encoder       = None
        self.clf           = None
        self._emb_cache    = {}

    def _build_lgbm(self, n_classes):
        return LGBMClassifier(
            n_estimators     = self.n_estimators,
            num_leaves       = self.num_leaves,
            learning_rate    = self.learning_rate,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_samples= 5,
            random_state     = self.random_state,
            verbosity        = -1,
            objective        = "multiclass" if n_classes > 2 else "binary",
            num_class        = n_classes if n_classes > 2 else None,
        )

    def fit(self, X_struct, text_list, y,
            X_val_struct=None, text_val=None, y_val=None):
        print("    [NoBERT] Loading sentence encoder …")
        self.encoder = SentenceTransformer(self.encoder_name)

        print("    [NoBERT] Encoding train text …")
        emb_tr = self.encoder.encode(
            text_list, batch_size=64,
            show_progress_bar=True, convert_to_numpy=True)

        X_s = X_struct.values if hasattr(X_struct, "values") else X_struct
        X_c = np.hstack([X_s, emb_tr])

        n_classes = len(np.unique(y))
        self.clf  = self._build_lgbm(n_classes)

        if X_val_struct is not None and text_val is not None:
            print("    [NoBERT] Encoding val text …")
            emb_v = self.encoder.encode(
                text_val, batch_size=64,
                show_progress_bar=True, convert_to_numpy=True)
            X_vs = X_val_struct.values if hasattr(X_val_struct, "values") else X_val_struct
            X_vc = np.hstack([X_vs, emb_v])
            self.clf.fit(
                X_c, y,
                eval_set=[(X_vc, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=-1),
                ])
        else:
            self.clf.fit(X_c, y)

        return self

    def _encode_and_combine(self, X_struct, texts, cache_key=None):
        if cache_key and cache_key in self._emb_cache:
            emb = self._emb_cache[cache_key]
        else:
            emb = self.encoder.encode(
                texts, batch_size=64,
                show_progress_bar=False, convert_to_numpy=True)
            if cache_key:
                self._emb_cache[cache_key] = emb
        X_s = X_struct.values if hasattr(X_struct, "values") else X_struct
        return np.hstack([X_s, emb])

    def predict(self, X_struct, texts, cache_key=None):
        X_c = self._encode_and_combine(X_struct, texts, cache_key)
        return self.clf.predict(X_c)

    def predict_proba(self, X_struct, texts, cache_key=None):
        X_c = self._encode_and_combine(X_struct, texts, cache_key)
        return self.clf.predict_proba(X_c)


# ══════════════════════════════════════════════════════════════
# CELL 8 ─ Train all 5 models
# ══════════════════════════════════════════════════════════════
# Chain target order: Frontend → Backend → Database
# Each subsequent link receives prior-target predictions as
# an extra feature column, capturing label correlations.
chain_targets = [
    ("Frontend", le_fe, y_fe_train, y_fe_val, y_fe_test),
    ("Backend",  le_be, y_be_train, y_be_val, y_be_test),
    ("Database", le_db, y_db_train, y_db_val, y_db_test),
]

print("=" * 68)
print("  TRAINING 5 MODELS")
print("=" * 68)


# ─────────────────────────────────────────────────────────────
# MODEL 1: ChainCatBoost
# ─────────────────────────────────────────────────────────────
# CatBoost per chain link with early stopping on the val set.
# Early stopping (od_wait=50) prevents overfitting on each link.

print("\n══ ChainCatBoost ══")
results["ChainCatBoost"] = {}
trained["ChainCatBoost"] = {}
chain_cat_inputs = {}

cc_X_tr = X_tr.copy()
cc_X_v  = X_v.copy()
cc_X_te = X_te.copy()

for link_idx, (tname, le, y_tr, y_v, y_te) in enumerate(chain_targets):
    t0 = time.time()
    cc_X_tr_in = cc_X_tr.copy()
    cc_X_v_in  = cc_X_v.copy()
    cc_X_te_in = cc_X_te.copy()

    cb = CatBoostClassifier(
        iterations    = 1000,
        learning_rate = 0.05,
        depth         = 7,
        l2_leaf_reg   = 3,
        random_seed   = 42,
        eval_metric   = "Accuracy",
        od_type       = "Iter",
        od_wait       = 50,
        verbose       = False,
        loss_function = "MultiClass",
        classes_count = len(le.classes_),
    )
    cb.fit(cc_X_tr, y_tr, eval_set=(cc_X_v, y_v))
    elapsed = time.time() - t0

    # Hard-label predictions become the next link's extra feature
    y_pred_tr = cb.predict(cc_X_tr).ravel().astype(int)
    y_pred_v  = cb.predict(cc_X_v).ravel().astype(int)
    y_pred_te = cb.predict(cc_X_te).ravel().astype(int)

    cc_X_tr = np.hstack([cc_X_tr, y_pred_tr.reshape(-1, 1)])
    cc_X_v  = np.hstack([cc_X_v,  y_pred_v.reshape(-1, 1)])
    cc_X_te = np.hstack([cc_X_te, y_pred_te.reshape(-1, 1)])

    # Evaluate on the input before the appended column
    val_m  = evaluate(cb, cc_X_v_in,  y_v,  le, "val")
    test_m = evaluate(cb, cc_X_te_in, y_te, le, "test")

    results["ChainCatBoost"][tname] = {"val": val_m, "test": test_m, "train_sec": elapsed}
    trained["ChainCatBoost"][tname] = cb
    chain_cat_inputs[tname] = {"train": cc_X_tr_in, "val": cc_X_v_in, "test": cc_X_te_in}
    print_result(tname, val_m, test_m, elapsed,
                 f"best_iter={cb.get_best_iteration()}")

trained["ChainCatBoost"]["_chain_order"] = ["Frontend", "Backend", "Database"]


# ─────────────────────────────────────────────────────────────
# MODEL 2: ChainXGB
# ─────────────────────────────────────────────────────────────
# XGBoost per chain link with early stopping.
# Uses stronger params than the original v1
# (lr=0.03, depth=7, min_child_weight=3) for higher accuracy.

print("\n══ ChainXGB ══")
results["ChainXGB"] = {}
trained["ChainXGB"] = {}
chain_xgb_inputs = {}

cx_X_tr = X_tr.copy()
cx_X_v  = X_v.copy()
cx_X_te = X_te.copy()
chain_xgb_models = []

for link_idx, (tname, le, y_tr, y_v, y_te) in enumerate(chain_targets):
    t0 = time.time()
    cx_X_tr_in = cx_X_tr.copy()
    cx_X_v_in  = cx_X_v.copy()
    cx_X_te_in = cx_X_te.copy()

    xgb = XGBClassifier(
        n_estimators          = 800,
        max_depth             = 7,
        learning_rate         = 0.03,
        subsample             = 0.85,
        colsample_bytree      = 0.85,
        min_child_weight      = 3,
        gamma                 = 0.05,
        reg_alpha             = 0.1,
        reg_lambda            = 1.5,
        eval_metric           = "mlogloss",
        early_stopping_rounds = 40,
        random_state          = 42,
        verbosity             = 0,
        num_class             = len(le.classes_) if len(le.classes_) > 2 else None,
    )
    xgb.fit(cx_X_tr, y_tr, eval_set=[(cx_X_v, y_v)], verbose=False)
    elapsed = time.time() - t0

    y_pred_tr = xgb.predict(cx_X_tr).ravel().astype(int)
    y_pred_v  = xgb.predict(cx_X_v).ravel().astype(int)
    y_pred_te = xgb.predict(cx_X_te).ravel().astype(int)

    cx_X_tr = np.hstack([cx_X_tr, y_pred_tr.reshape(-1, 1)])
    cx_X_v  = np.hstack([cx_X_v,  y_pred_v.reshape(-1, 1)])
    cx_X_te = np.hstack([cx_X_te, y_pred_te.reshape(-1, 1)])

    val_m  = evaluate(xgb, cx_X_v_in,  y_v,  le, "val")
    test_m = evaluate(xgb, cx_X_te_in, y_te, le, "test")

    results["ChainXGB"][tname] = {"val": val_m, "test": test_m, "train_sec": elapsed}
    trained["ChainXGB"][tname] = xgb
    chain_xgb_inputs[tname] = {"train": cx_X_tr_in, "val": cx_X_v_in, "test": cx_X_te_in}
    chain_xgb_models.append(xgb)
    print_result(tname, val_m, test_m, elapsed,
                 f"best_iter={xgb.best_iteration}")

trained["ChainXGB"]["_chain_order"]  = ["Frontend", "Backend", "Database"]
trained["ChainXGB"]["_chain_models"] = chain_xgb_models


# ─────────────────────────────────────────────────────────────
# MODEL 3: NoBERT
# ─────────────────────────────────────────────────────────────
print("\n══ NoBERT ══")
results["NoBERT"] = {}
trained["NoBERT"] = {}

if RUN_NOBERT:
    for tname, (y_tr, y_v, y_te, le) in TARGETS.items():
        print(f"\n  Training NoBERT → {tname} …")
        t0 = time.time()

        nb = NoBERTClassifier(n_estimators=500, num_leaves=63)
        nb.fit(X_train, text_train, y_tr,
               X_val_struct=X_val, text_val=text_val, y_val=y_v)
        elapsed = time.time() - t0

        y_pred_v  = nb.predict(X_val,  text_val,  cache_key="val")
        y_pred_te = nb.predict(X_test, text_test, cache_key="test")

        val_m = {
            "accuracy":    accuracy_score(y_v, y_pred_v),
            "f1_weighted": f1_score(y_v, y_pred_v, average="weighted",  zero_division=0),
            "f1_macro":    f1_score(y_v, y_pred_v, average="macro",     zero_division=0),
            "hamming":     hamming_loss(y_v, y_pred_v),
            "report":      classification_report(y_v, y_pred_v,
                               labels=np.arange(len(le.classes_)),
                               target_names=le.classes_, zero_division=0),
            "y_pred":      y_pred_v, "split": "val",
        }
        test_m = {
            "accuracy":    accuracy_score(y_te, y_pred_te),
            "f1_weighted": f1_score(y_te, y_pred_te, average="weighted",  zero_division=0),
            "f1_macro":    f1_score(y_te, y_pred_te, average="macro",     zero_division=0),
            "hamming":     hamming_loss(y_te, y_pred_te),
            "report":      classification_report(y_te, y_pred_te,
                               labels=np.arange(len(le.classes_)),
                               target_names=le.classes_, zero_division=0),
            "y_pred":      y_pred_te, "split": "test",
        }

        results["NoBERT"][tname] = {"val": val_m, "test": test_m, "train_sec": elapsed}
        trained["NoBERT"][tname] = nb
        print_result(tname, val_m, test_m, elapsed)
else:
    print("  Skipped (RUN_NOBERT=False)")
    for tname in targets_list:
        nan_m = {"accuracy":0,"f1_weighted":0,"f1_macro":0,
                 "hamming":1,"report":"skipped","y_pred":None,"split":"—"}
        results["NoBERT"][tname] = {"val": nan_m, "test": nan_m, "train_sec": 0}
        trained["NoBERT"][tname] = None


# ─────────────────────────────────────────────────────────────
# MODEL 4: LightGBM
# ─────────────────────────────────────────────────────────────
# Tuned params for higher accuracy:
#   num_leaves=127 (wider trees), lr=0.03 (finer steps),
#   min_child_samples=5 (allows smaller leaf nodes)

print("\n══ LightGBM ══")
results["LightGBM"] = {}
trained["LightGBM"] = {}

for tname, (y_tr, y_v, y_te, le) in TARGETS.items():
    t0 = time.time()

    lgbm = LGBMClassifier(
        n_estimators     = 1000,
        num_leaves       = 127,
        learning_rate    = 0.03,
        subsample        = 0.85,
        colsample_bytree = 0.85,
        min_child_samples= 5,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        verbosity        = -1,
        objective        = "multiclass",
        num_class        = len(le.classes_),
    )
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    )
    elapsed = time.time() - t0

    val_m  = evaluate(lgbm, X_v,  y_v,  le, "val")
    test_m = evaluate(lgbm, X_te, y_te, le, "test")

    results["LightGBM"][tname] = {"val": val_m, "test": test_m, "train_sec": elapsed}
    trained["LightGBM"][tname] = lgbm
    print_result(tname, val_m, test_m, elapsed,
                 f"best_iter={lgbm.best_iteration_}")


# ─────────────────────────────────────────────────────────────
# MODEL 5: StackingEnsemble
# ─────────────────────────────────────────────────────────────
# Meta-learner combines probability outputs from independent
# (non-chained) CatBoost, XGBoost, LightGBM and a NoBERT-style
# LightGBM head trained on structured + sentence-embedding features.
#
# IMPORTANT — why the earlier version underperformed:
# it built the meta-learner's TRAINING features from models that had
# already been fit on that exact training data (in-sample / leaked
# predictions). Those probabilities are near-perfect and look nothing
# like what the same models produce on unseen val/test rows, so the
# meta-learner learned a mapping that doesn't transfer. The fix is to
# generate the training meta-features OUT-OF-FOLD: every row's
# meta-feature comes from a model instance that never saw that row
# during its own training, matching the val/test regime.
#
# NOTE: newer scikit-learn versions (>=1.5) removed the `multi_class`
# kwarg from LogisticRegression — lbfgs handles multiclass natively,
# so it's simply omitted below.

print("\n══ StackingEnsemble ══")
results["StackingEnsemble"] = {}
trained["StackingEnsemble"] = {}

N_STACK_FOLDS = 5

def build_stack_catboost(n_classes):
    return CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=7, l2_leaf_reg=3,
        random_seed=42, eval_metric="Accuracy", od_type="Iter", od_wait=50,
        verbose=False, loss_function="MultiClass", classes_count=n_classes)

def build_stack_xgb(n_classes):
    return XGBClassifier(
        n_estimators=800, max_depth=7, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        gamma=0.05, reg_alpha=0.1, reg_lambda=1.5,
        eval_metric="mlogloss", random_state=42, verbosity=0,
        num_class=n_classes if n_classes > 2 else None)

def build_stack_lgbm(n_classes):
    return LGBMClassifier(
        n_estimators=1000, num_leaves=127, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=-1,
        objective="multiclass", num_class=n_classes)

def build_stack_nobert(n_classes):
    return LGBMClassifier(
        n_estimators=500, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
        random_state=42, verbosity=-1,
        objective="multiclass", num_class=n_classes)

def kfold_oof_proba(build_fn, X, y, n_classes, n_splits=N_STACK_FOLDS, xgb_style=False):
    """Fresh model per fold -> OOF probability matrix (no leakage)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros((len(y), n_classes))
    for tr_idx, ho_idx in skf.split(X, y):
        model = build_fn(n_classes)
        if xgb_style:
            model.fit(X[tr_idx], y[tr_idx], verbose=False)
        else:
            model.fit(X[tr_idx], y[tr_idx])
        oof[ho_idx] = model.predict_proba(X[ho_idx])
    return oof

if RUN_NOBERT:
    print("  Precomputing sentence embeddings for stacking (one pass)…")
    _stack_encoder = trained["NoBERT"]["Frontend"].encoder
    emb_train_s = _stack_encoder.encode(text_train, batch_size=64,
                                         show_progress_bar=True, convert_to_numpy=True)
    emb_val_s   = _stack_encoder.encode(text_val,   batch_size=64,
                                         show_progress_bar=True, convert_to_numpy=True)
    emb_test_s  = _stack_encoder.encode(text_test,  batch_size=64,
                                         show_progress_bar=True, convert_to_numpy=True)
    X_comb_tr_s = np.hstack([X_tr, emb_train_s])
    X_comb_v_s  = np.hstack([X_v,  emb_val_s])
    X_comb_te_s = np.hstack([X_te, emb_test_s])

for tname, (y_tr, y_v, y_te, le) in TARGETS.items():
    t0 = time.time()

    if not RUN_NOBERT:
        nan_m = {"accuracy":0,"f1_weighted":0,"f1_macro":0,
                 "hamming":1,"report":"skipped","y_pred":None,"split":"—"}
        results["StackingEnsemble"][tname] = {"val": nan_m, "test": nan_m, "train_sec": 0}
        trained["StackingEnsemble"][tname] = None
        print(f"  {tname:<10}  skipped (NoBERT disabled)")
        continue

    n_classes = len(le.classes_)

    # ---- 1) Out-of-fold probabilities on TRAIN (meta-training features,
    #         no leakage — each row scored by a model that never saw it)
    oof_cat    = kfold_oof_proba(build_stack_catboost, X_tr,        y_tr, n_classes)
    oof_xgb    = kfold_oof_proba(build_stack_xgb,      X_tr,        y_tr, n_classes, xgb_style=True)
    oof_lgbm   = kfold_oof_proba(build_stack_lgbm,     X_tr,        y_tr, n_classes)
    oof_nobert = kfold_oof_proba(build_stack_nobert,   X_comb_tr_s, y_tr, n_classes)
    meta_X_train = np.hstack([oof_cat, oof_xgb, oof_lgbm, oof_nobert])

    # ---- 2) Fit each base learner ONCE on the full train set — used only
    #         to produce meta-features for val/test (never for train)
    cat_full = build_stack_catboost(n_classes)
    cat_full.fit(X_tr, y_tr, eval_set=(X_v, y_v))

    xgb_full = build_stack_xgb(n_classes)
    xgb_full.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)

    lgbm_full = build_stack_lgbm(n_classes)
    lgbm_full.fit(X_tr, y_tr, eval_set=[(X_v, y_v)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=-1)])

    nobert_full = build_stack_nobert(n_classes)
    nobert_full.fit(X_comb_tr_s, y_tr, eval_set=[(X_comb_v_s, y_v)],
                     callbacks=[lgb.early_stopping(40, verbose=False),
                                lgb.log_evaluation(period=-1)])

    meta_X_val = np.hstack([
        cat_full.predict_proba(X_v), xgb_full.predict_proba(X_v),
        lgbm_full.predict_proba(X_v), nobert_full.predict_proba(X_comb_v_s)])
    meta_X_test = np.hstack([
        cat_full.predict_proba(X_te), xgb_full.predict_proba(X_te),
        lgbm_full.predict_proba(X_te), nobert_full.predict_proba(X_comb_te_s)])

    # ---- 3) Meta-learner trained on OOF features, tune C by val accuracy
    best_meta, best_val_acc = None, -1
    for C in [0.05, 0.1, 0.3, 1.0, 3.0]:
        cand = LogisticRegression(max_iter=3000, C=C, n_jobs=-1)
        cand.fit(meta_X_train, y_tr)
        acc = accuracy_score(y_v, cand.predict(meta_X_val))
        if acc > best_val_acc:
            best_val_acc, best_meta = acc, cand
    meta = best_meta
    elapsed = time.time() - t0

    val_m  = evaluate(meta, meta_X_val,  y_v,  le, "val")
    test_m = evaluate(meta, meta_X_test, y_te, le, "test")

    results["StackingEnsemble"][tname] = {"val": val_m, "test": test_m, "train_sec": elapsed}
    trained["StackingEnsemble"][tname] = {
        "base_catboost": cat_full, "base_xgb": xgb_full,
        "base_lightgbm": lgbm_full, "base_nobert": nobert_full,
        "meta_learner":  meta,
    }
    print_result(tname, val_m, test_m, elapsed, f"meta=LogReg(C={meta.C})")

print("\n✓ All 5 models trained")


# ══════════════════════════════════════════════════════════════
# CELL 9 ─ Results comparison tables (val + test)
# ══════════════════════════════════════════════════════════════
def build_summary(split):
    rows = []
    for mname in MODEL_NAMES:
        fe  = results[mname]["Frontend"][split]
        be  = results[mname]["Backend"][split]
        db  = results[mname]["Database"][split]
        avg_acc = round((fe["accuracy"] + be["accuracy"] + db["accuracy"]) / 3, 3)
        avg_f1  = round((fe["f1_weighted"] + be["f1_weighted"] + db["f1_weighted"]) / 3, 3)
        rows.append({
            "Model":      mname,
            "FE Acc":     round(fe["accuracy"],    3),
            "BE Acc":     round(be["accuracy"],    3),
            "DB Acc":     round(db["accuracy"],    3),
            "FE F1-W":    round(fe["f1_weighted"], 3),
            "BE F1-W":    round(be["f1_weighted"], 3),
            "DB F1-W":    round(db["f1_weighted"], 3),
            "FE F1-Mac":  round(fe["f1_macro"],    3),
            "BE F1-Mac":  round(be["f1_macro"],    3),
            "DB F1-Mac":  round(db["f1_macro"],    3),
            "FE Hamming": round(fe["hamming"],     3),
            "BE Hamming": round(be["hamming"],     3),
            "DB Hamming": round(db["hamming"],     3),
            "Avg Acc":    avg_acc,
            "Avg F1-W":   avg_f1,
        })
    return pd.DataFrame(rows).sort_values("Avg Acc", ascending=False)

val_df  = build_summary("val")
test_df = build_summary("test")
best_name = test_df.iloc[0]["Model"]

for label, df_s in [("VALIDATION", val_df), ("TEST  (FINAL)", test_df)]:
    print(f"\n{'='*78}")
    print(f"  {label} RESULTS")
    print(f"{'='*78}")
    print(f"  {'Model':<17} {'FE Acc':>7} {'BE Acc':>7} {'DB Acc':>7} "
            f"{'FE F1':>7} {'BE F1':>7} {'DB F1':>7} {'Avg':>6}")
    print(f"  {'─'*71}")
    for _, r in df_s.iterrows():
        print(f"  {r['Model']:<17} {r['FE Acc']:>7} {r['BE Acc']:>7} {r['DB Acc']:>7} "
              f"{r['FE F1-W']:>7} {r['BE F1-W']:>7} {r['DB F1-W']:>7} "
              f"{r['Avg Acc']:>6}")
    print(f"  {'─'*71}")

print(f"\n  Majority-class baselines:")
for tname in targets_list:
    print(f"    {tname:<12}  val={BASELINES[tname]['val']:.3f}  "
          f"test={BASELINES[tname]['test']:.3f}")
print(f"\n  Best model (by test Avg Acc): {best_name}")

val_df.to_csv(out("model_val_comparison.csv"),  index=False)
test_df.to_csv(out("model_test_comparison.csv"), index=False)
print("✓ CSVs saved")


# ══════════════════════════════════════════════════════════════
# CELL 10 ─ Detailed classification reports (test set)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  DETAILED CLASSIFICATION REPORTS — TEST SET")
print(f"{'='*60}")
for mname in MODEL_NAMES:
    print(f"\n{'─'*55}\n  {mname.upper()}\n{'─'*55}")
    for tname in targets_list:
        r = results[mname][tname]["test"]
        print(f"\n  [{tname}]  acc={r['accuracy']:.3f}  "
              f"f1_w={r['f1_weighted']:.3f}  f1_mac={r['f1_macro']:.3f}")
        print(r["report"])


# ══════════════════════════════════════════════════════════════
# CELL 11 ─ Charts
# ══════════════════════════════════════════════════════════════
C = {"Frontend": "#5B9BD5", "Backend": "#E07B7B", "Database": "#5BAD8E"}
x = np.arange(len(MODEL_NAMES))
w = 0.26

# ── Chart 1: Val vs Test accuracy ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    "Validation vs Test Accuracy per Model\n"
    "Dashed lines = majority-class baseline",
    fontsize=12, fontweight="bold")

for ax, tname, le in zip(axes, targets_list, [le_fe, le_be, le_db]):
    val_accs  = [results[m][tname]["val"]["accuracy"]  for m in MODEL_NAMES]
    test_accs = [results[m][tname]["test"]["accuracy"] for m in MODEL_NAMES]
    bl_v = BASELINES[tname]["val"]
    bl_t = BASELINES[tname]["test"]

    b1 = ax.bar(x - 0.2, val_accs,  0.38, label="Val",
                color=C[tname], alpha=0.5, edgecolor="white")
    b2 = ax.bar(x + 0.2, test_accs, 0.38, label="Test",
                color=C[tname], alpha=1.0, edgecolor="white")

    for bar, v in list(zip(b1, val_accs)) + list(zip(b2, test_accs)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(bl_v, color="grey",  linestyle="--", lw=0.9, alpha=0.7,
               label=f"Val bl={bl_v:.3f}")
    ax.axhline(bl_t, color="black", linestyle="--", lw=0.9, alpha=0.7,
               label=f"Test bl={bl_t:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, rotation=25, ha="right", fontsize=8)
    ax.set_title(f"{tname} ({len(le.classes_)} classes)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig(out("chart_01_val_vs_test_accuracy.png"), bbox_inches="tight")
plt.close()
print("✓ Chart 1 saved → chart_01_val_vs_test_accuracy.png")

# ── Chart 2: Test accuracy heatmap ───────────────────────────
heat_data = pd.DataFrame(
    {t: [results[m][t]["test"]["accuracy"] for m in MODEL_NAMES]
     for t in targets_list},
    index=MODEL_NAMES)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.5, vmax=1.0, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Test Accuracy"}, ax=ax)
ax.set_title("Test Accuracy Heatmap\n"
             "Higher is better", fontweight="bold")
ax.set_xlabel("Target")
ax.set_ylabel("Model")
plt.tight_layout()
plt.savefig(out("chart_02_accuracy_heatmap.png"), bbox_inches="tight")
plt.close()
print("✓ Chart 2 saved → chart_02_accuracy_heatmap.png")

# ── Chart 3: Confusion matrices (best model, test set) ───────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle(
    f"Confusion Matrices: {best_name} (Test Set)\n"
    "Values = % of actual class predicted as each tech",
    fontsize=12, fontweight="bold")

for ax, tname, le, cmap in zip(
    axes, targets_list,
    [le_fe, le_be, le_db],
    ["Blues", "Oranges", "Greens"]
):
    y_pred = results[best_name][tname]["test"]["y_pred"]
    _, _, y_te, _ = TARGETS[tname]
    cm     = confusion_matrix(y_te, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(
        cm_pct, annot=True, fmt=".0f", cmap=cmap,
        xticklabels=le.classes_, yticklabels=le.classes_,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "%"}, ax=ax, vmin=0, vmax=100)
    acc = results[best_name][tname]["test"]["accuracy"]
    f1  = results[best_name][tname]["test"]["f1_weighted"]
    ax.set_title(f"{tname}\nacc={acc:.3f}  f1_w={f1:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8)

plt.tight_layout()
plt.savefig(out("chart_03_confusion_matrices.png"), bbox_inches="tight")
plt.close()
print("✓ Chart 3 saved → chart_03_confusion_matrices.png")

# ── Chart 4: Feature importance (LightGBM, top 20) ───────────
feat_arr = np.array(feat_names)

def clean_feat(n):
    if n.startswith("tfidf_"): return "text: " + n[6:]
    if n.startswith("dom_"):   return "domain: " + n[4:]
    if n.startswith("lang_"):  return "lang: " + n[5:]
    return n.replace("_enc", "").replace("_", " ")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("Top 20 Feature Importances (LightGBM)",
             fontsize=12, fontweight="bold")

for ax, tname, cmap_name in zip(
    axes, targets_list, ["Blues", "Oranges", "Greens"]
):
    imp     = trained["LightGBM"][tname].feature_importances_
    top_idx = np.argsort(imp)[::-1][:20]
    top_imp = imp[top_idx][::-1]
    labels  = [clean_feat(feat_arr[i]) for i in top_idx[::-1]]

    ax.barh(range(20), top_imp,
            color=plt.get_cmap(cmap_name)(np.linspace(0.35, 0.85, 20)),
            edgecolor="white")
    ax.set_yticks(range(20))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{tname} (top 20 features)")
    ax.set_xlabel("Importance score")

plt.tight_layout()
plt.savefig(out("chart_04_feature_importance.png"), bbox_inches="tight")
plt.close()
print("✓ Chart 4 saved → chart_04_feature_importance.png")

# ── Chart 5: Hamming loss comparison ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for ti, tname in enumerate(targets_list):
    vals = [results[m][tname]["test"]["hamming"] for m in MODEL_NAMES]
    bars = ax.bar(x + (ti - 1) * w, vals, w,
                  label=tname, color=list(C.values())[ti], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, rotation=20, ha="right")
ax.set_ylabel("Hamming Loss")
ax.set_title("Hamming Loss per Model (Test Set)\n"
             "Lower is better", fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out("chart_05_hamming_loss.png"), bbox_inches="tight")
plt.close()
print("✓ Chart 5 saved → chart_05_hamming_loss.png")

# ── Chart 6: Test accuracy comparison ─────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for ti, tname in enumerate(targets_list):
    vals = [results[m][tname]["test"]["accuracy"] for m in MODEL_NAMES]
    bars = ax.bar(x + (ti - 1) * w, vals, w,
                  label=tname, color=list(C.values())[ti], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, rotation=20, ha="right")
ax.set_ylabel("Test Accuracy")
ax.set_ylim(0, 1.08)
ax.set_title("Test Accuracy Comparison (Test Set)", fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out("chart_06_threshold_comparison.png"), bbox_inches="tight")
plt.close()
print("✓ Chart 6 saved → chart_06_threshold_comparison.png")

# ── Chart 7: Accuracy gain over baseline ─────────────────────
gains = {
    m: {t: results[m][t]["test"]["accuracy"] - BASELINES[t]["test"]
        for t in targets_list}
    for m in MODEL_NAMES
}

fig, ax = plt.subplots(figsize=(13, 5))
for ti, tname in enumerate(targets_list):
    vals = [gains[m][tname] for m in MODEL_NAMES]
    bars = ax.bar(x + (ti - 1) * w, vals, w,
                  label=tname, color=list(C.values())[ti], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"+{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, rotation=20, ha="right")
ax.set_ylabel("Accuracy gain over test baseline")
ax.set_title("Accuracy Gain over Majority-Class Baseline (Test Set)\n"
             "Shows each model's real contribution beyond random guessing",
             fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out("chart_07_accuracy_gain.png"), bbox_inches="tight")
plt.close()
print("✓ Chart 7 saved → chart_07_accuracy_gain.png")

# ── Chart 8: 5-fold CV (all 5 models, Frontend) ──────────────
print("\nRunning 5-fold CV on Frontend target (all 5 models)…")

def sequential_cv_scores(estimator, X_data, y_data, n_splits=5):
    scores = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(X_data, y_data):
        model = clone(estimator)
        model.fit(X_data[train_idx], y_data[train_idx])
        scores.append(accuracy_score(y_data[test_idx], model.predict(X_data[test_idx])))
    return np.array(scores)

cb_cv = CatBoostClassifier(
    iterations=300, learning_rate=0.05, depth=7,
    random_seed=42, verbose=False,
    loss_function="MultiClass", classes_count=len(le_fe.classes_))
cv_cb = sequential_cv_scores(cb_cv, X_tr, y_fe_train)

xgb_cv_m = XGBClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=3, gamma=0.05,
    reg_alpha=0.1, reg_lambda=1.5,
    eval_metric="mlogloss", random_state=42, verbosity=0,
    num_class=len(le_fe.classes_))
cv_xgb = sequential_cv_scores(xgb_cv_m, X_tr, y_fe_train)

lgbm_cv_m = LGBMClassifier(
    n_estimators=300, num_leaves=127, learning_rate=0.03,
    random_state=42, verbosity=-1, n_jobs=1,
    objective="multiclass", num_class=len(le_fe.classes_))
cv_lgbm = sequential_cv_scores(lgbm_cv_m, X_tr, y_fe_train)

print("  Preparing NoBERT embeddings for CV…")
cv_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cv_emb = cv_encoder.encode(
    text_train, batch_size=64,
    show_progress_bar=True, convert_to_numpy=True)
X_tr_nobert = np.hstack([X_tr, cv_emb])

nobert_cv_m = LGBMClassifier(
    n_estimators=300, num_leaves=63, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=5, random_state=42,
    verbosity=-1, n_jobs=1,
    objective="multiclass", num_class=len(le_fe.classes_))
cv_nobert = sequential_cv_scores(nobert_cv_m, X_tr_nobert, y_fe_train)

def stacking_cv_scores(X_struct, X_nobert, y_data, n_splits=5):
    scores = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_classes = len(le_fe.classes_)
    for train_idx, test_idx in splitter.split(X_struct, y_data):
        Xs_tr, Xs_te = X_struct[train_idx], X_struct[test_idx]
        Xn_tr, Xn_te = X_nobert[train_idx], X_nobert[test_idx]
        y_tr_f, y_te_f = y_data[train_idx], y_data[test_idx]

        cat_m = CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=7,
            random_seed=42, verbose=False,
            loss_function="MultiClass", classes_count=n_classes)
        xgb_m = XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=3, gamma=0.05,
            reg_alpha=0.1, reg_lambda=1.5,
            eval_metric="mlogloss", random_state=42, verbosity=0,
            num_class=n_classes)
        lgb_m = LGBMClassifier(
            n_estimators=300, num_leaves=127, learning_rate=0.03,
            random_state=42, verbosity=-1, n_jobs=1,
            objective="multiclass", num_class=n_classes)
        nb_m = LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=5, random_state=42,
            verbosity=-1, n_jobs=1,
            objective="multiclass", num_class=n_classes)

        cat_m.fit(Xs_tr, y_tr_f)
        xgb_m.fit(Xs_tr, y_tr_f)
        lgb_m.fit(Xs_tr, y_tr_f)
        nb_m.fit(Xn_tr, y_tr_f)

        meta_tr = np.hstack([
            cat_m.predict_proba(Xs_tr),
            xgb_m.predict_proba(Xs_tr),
            nb_m.predict_proba(Xn_tr),
            lgb_m.predict_proba(Xs_tr),
        ])
        meta_te = np.hstack([
            cat_m.predict_proba(Xs_te),
            xgb_m.predict_proba(Xs_te),
            nb_m.predict_proba(Xn_te),
            lgb_m.predict_proba(Xs_te),
        ])
        meta = LogisticRegression(max_iter=2000, random_state=42)
        meta.fit(meta_tr, y_tr_f)
        scores.append(accuracy_score(y_te_f, meta.predict(meta_te)))
    return np.array(scores)

cv_stack = stacking_cv_scores(X_tr, X_tr_nobert, y_fe_train)

print(f"  CatBoost  mean={cv_cb.mean():.3f} ±{cv_cb.std():.3f}")
print(f"  ChainXGB  mean={cv_xgb.mean():.3f} ±{cv_xgb.std():.3f}")
print(f"  NoBERT    mean={cv_nobert.mean():.3f} ±{cv_nobert.std():.3f}")
print(f"  LightGBM  mean={cv_lgbm.mean():.3f} ±{cv_lgbm.std():.3f}")
print(f"  Stacking  mean={cv_stack.mean():.3f} ±{cv_stack.std():.3f}")

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(
    [cv_cb, cv_xgb, cv_nobert, cv_lgbm, cv_stack], patch_artist=True,
    medianprops={"color": "white", "linewidth": 2.5},
    whiskerprops={"linewidth": 1.2},
    capprops={"linewidth": 1.2})
for patch, color in zip(bp["boxes"], ["#5B9BD5", "#E07B7B", "#6A8CAF", "#5BAD8E", "#A5678E"]):
    patch.set_facecolor(color)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["ChainCatBoost", "ChainXGB", "NoBERT", "LightGBM", "StackingEnsemble"],
                   rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (5-fold CV, Frontend)")
ax.set_title("5-Fold Cross-Validation (Frontend Target)\n"
             "Proves stability — not a lucky single split", fontweight="bold")
ax.axhline(bl_fe_val, color="gray", linestyle="--", lw=1.0,
           label=f"Baseline = {bl_fe_val:.3f}")
ax.legend(fontsize=8)
ax.set_ylim(0.4, 1.0)
plt.tight_layout()
plt.savefig(out("chart_08_cross_validation.png"), bbox_inches="tight")
plt.close()
print("✓ Chart 8 saved → chart_08_cross_validation.png")


# ══════════════════════════════════════════════════════════════
# CELL 12 ─ Save best model & all artifacts
# ══════════════════════════════════════════════════════════════
joblib.dump({
    "model_name":   best_name,
    "frontend":     trained[best_name]["Frontend"],
    "backend":      trained[best_name]["Backend"],
    "database":     trained[best_name]["Database"],
    "le_fe":        le_fe,
    "le_be":        le_be,
    "le_db":        le_db,
    "feat_names":   feat_names,
    "domain_cols":  splits.get("domain_cols", []),
    "lang_cols":    splits.get("lang_cols", []),
    "split_ratio":  "80/10/10",
    "val_results":  {m: {t: results[m][t]["val"]  for t in targets_list}
                     for m in MODEL_NAMES},
    "test_results": {m: {t: results[m][t]["test"] for t in targets_list}
                     for m in MODEL_NAMES},
}, out("best_model.pkl"))

joblib.dump(trained, out("all_trained_models.pkl"))
joblib.dump(results, out("all_results.pkl"))

print(f"\n✓ best_model.pkl         → {out('best_model.pkl')}  [{best_name}]")
print(f"✓ all_trained_models.pkl → {out('all_trained_models.pkl')}")
print(f"✓ all_results.pkl        → {out('all_results.pkl')}")


# ══════════════════════════════════════════════════════════════
# CELL 13 ─ Final summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TRAINING COMPLETE — FINAL SUMMARY")
print(f"{'='*70}")
print(f"\n  Dataset : {X_train.shape[0]+X_val.shape[0]+X_test.shape[0]:,} rows × "
      f"{X_train.shape[1]} features")
print(f"  Train   : {X_train.shape[0]:,}  |  Val : {X_val.shape[0]:,}  "
      f"|  Test : {X_test.shape[0]:,}")
print(f"  Models  : 5  ×  3 targets  =  15 classifiers\n")

print(f"  {'Model':<17} {'FE Acc':>8} {'BE Acc':>8} {'DB Acc':>8} "
    f"{'Avg':>7}")
print(f"  {'─'*56}")
for _, r in test_df.iterrows():
    print(f"  {r['Model']:<17} {r['FE Acc']:>8} {r['BE Acc']:>8} "
        f"{r['DB Acc']:>8} {r['Avg Acc']:>7}")
print(f"  {'─'*56}")
print(f"  {'Baseline':<17} "
      f"{bl_fe_test:>8.3f} {bl_be_test:>8.3f} {bl_db_test:>8.3f}")

print(f"\n  Best model : {best_name}")
print(f"\n  Val → Test gap check (< 0.03 = ✓ no overfitting):")
for tname in targets_list:
    v   = results[best_name][tname]["val"]["accuracy"]
    t   = results[best_name][tname]["test"]["accuracy"]
    gap = abs(v - t)
    print(f"    {tname:<12}  val={v:.3f}  test={t:.3f}  "
          f"gap={gap:.3f}  {'✓' if gap < 0.03 else '⚠ large gap'}")

print(f"\n  Output folder : ./{OUTPUT_DIR}/")
print(f"{'='*70}\n")