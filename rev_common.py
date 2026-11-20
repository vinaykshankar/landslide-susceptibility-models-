"""Shared helpers for the revision analyses (Frontiers R1/R2 comments)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(10))
BLOCK_M = 5000
MODELS = ["LR", "RF", "XGB", "LGBM"]

CONT = ["elevation", "slope", "northness", "eastness", "plan_curvature", "profile_curvature", "tri",
        "twi", "spi", "dist_streams", "rain_annual_mean", "rain_aug2018", "soil_clay", "soil_sand", "soil_bdod"]
TERRAIN = ["slope", "tri", "plan_curvature", "profile_curvature", "northness", "eastness", "twi", "spi", "dist_streams"]
# adversarial AUC from outputs/no_roads/tables/covariate_shift.csv; "stable" = AUC <= 0.55
STABLE_CONT = ["slope", "tri", "plan_curvature", "profile_curvature", "northness", "eastness", "twi", "spi",
               "dist_streams", "soil_clay", "soil_bdod"]


def load(tag="no_roads"):
    return pd.read_csv(PROC / f"samples_{tag}.csv")


def onehot(df):
    return ([c for c in df.columns if c.startswith("lc_") and c != "lc_class"]
            + [c for c in df.columns if c.startswith("lith_") and c != "lith_class"])


def feature_sets(df, with_roads=False):
    oh = onehot(df)
    rd = ["dist_roads"] if with_roads and "dist_roads" in df.columns else []
    return {
        "S1_all": CONT + rd + oh,
        "S3_no_event_rain": [f for f in CONT if f != "rain_aug2018"] + rd + oh,
        "S4_no_rainfall": [f for f in CONT if not f.startswith("rain_")] + rd + oh,
        "T1_terrain_only": TERRAIN,                      # slope-scale terrain + hydrology, no elevation
        "T2_terrain_elev": TERRAIN + ["elevation"],
        "T3_stable_only": STABLE_CONT + oh,              # all factors with adversarial AUC <= 0.55
    }


def models(seed, params=None):
    p = params or {}
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **p.get("LR", {"C": 1.0}))),
        "RF": RandomForestClassifier(n_jobs=2, random_state=seed,
                                     **p.get("RF", dict(n_estimators=500, min_samples_leaf=2, max_features="sqrt"))),
        "XGB": XGBClassifier(n_jobs=2, random_state=seed, eval_metric="logloss",
                             **p.get("XGB", dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                                 colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0))),
        "LGBM": LGBMClassifier(n_jobs=2, random_state=seed, verbose=-1,
                               **p.get("LGBM", dict(n_estimators=400, num_leaves=15, learning_rate=0.05, subsample=0.8,
                                                    subsample_freq=1, colsample_bytree=0.8, min_child_samples=10))),
    }


def blocks(d, m=BLOCK_M):
    return (np.floor(d.x / m).astype(int).astype(str) + "_" + np.floor(d.y / m).astype(int).astype(str)).values


def spatial_cv_pred(model, d, feats):
    return cross_val_predict(model, d[feats], d.label, cv=GroupKFold(5), groups=blocks(d), method="predict_proba")[:, 1]


def auc_subsets(d, p, cuts=(0, 10, 15, 20, 25)):
    """AUC on the whole sample and on increasingly steep subsets."""
    out = {}
    for c in cuts:
        m = d.slope.values >= c
        y = d.label.values[m]
        out[f"AUC_slope_ge_{c}"] = roc_auc_score(y, p[m]) if 0 < y.mean() < 1 and m.sum() > 30 else np.nan
        out[f"n_slope_ge_{c}"] = int(m.sum())
        out[f"pos_frac_slope_ge_{c}"] = float(y.mean()) if m.sum() else np.nan
    return out


def fdr_bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj
