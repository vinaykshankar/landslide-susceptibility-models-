"""R5: does hyperparameter tuning change the model ranking under transfer?
Random search selected by within-district spatial block CV, then applied to transfer.
Answers Reviewer 1 point 6 and Reviewer 2 point 3. -> outputs/revision/tuning_*.csv
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from rev_common import *

S = load("no_roads")
FE = feature_sets(S)["S1_all"]
N_CONF, TUNE_SEEDS = 20, [0, 1, 2]


def sample_conf(mname, rng):
    if mname == "LR":
        return dict(C=float(10 ** rng.uniform(-3, 2)))
    if mname == "RF":
        return dict(n_estimators=int(rng.choice([300, 500, 800])),
                    min_samples_leaf=int(rng.choice([1, 2, 4, 8])),
                    max_features=str(rng.choice(["sqrt", "log2"])),
                    max_depth=int(rng.choice([6, 10, 20, 30])))
    if mname == "XGB":
        return dict(n_estimators=int(rng.choice([200, 400, 800])), max_depth=int(rng.choice([2, 3, 4, 6, 8])),
                    learning_rate=float(10 ** rng.uniform(-2, -0.7)), subsample=float(rng.uniform(0.6, 1.0)),
                    colsample_bytree=float(rng.uniform(0.6, 1.0)), min_child_weight=int(rng.choice([1, 2, 5, 10])),
                    reg_lambda=float(10 ** rng.uniform(-1, 1.5)))
    return dict(n_estimators=int(rng.choice([200, 400, 800])), num_leaves=int(rng.choice([7, 15, 31, 63])),
                learning_rate=float(10 ** rng.uniform(-2, -0.7)), subsample=float(rng.uniform(0.6, 1.0)),
                subsample_freq=1, colsample_bytree=float(rng.uniform(0.6, 1.0)),
                min_child_samples=int(rng.choice([5, 10, 20, 40])), reg_lambda=float(10 ** rng.uniform(-1, 1.5)))


def build(mname, conf, seed):
    if mname == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **conf))
    if mname == "RF":
        return RandomForestClassifier(n_jobs=2, random_state=seed, **conf)
    if mname == "XGB":
        return XGBClassifier(n_jobs=2, random_state=seed, eval_metric="logloss", **conf)
    return LGBMClassifier(n_jobs=2, random_state=seed, verbose=-1, **conf)


rows, best_confs = [], []
for seed in TUNE_SEEDS:
    for src, tgt in (("wayanad", "kodagu"), ("kodagu", "wayanad")):
        d = S[(S.district == src) & (S.seed == seed)].reset_index(drop=True)
        e = S[(S.district == tgt) & (S.seed == seed)].reset_index(drop=True)
        for mname in MODELS:
            rng = np.random.default_rng(1234 + seed)
            trials = []
            for i in range(N_CONF):
                conf = sample_conf(mname, rng)
                try:
                    p = spatial_cv_pred(build(mname, conf, seed), d, FE)
                    trials.append((roc_auc_score(d.label, p), conf))
                except Exception as ex:
                    print("skip", mname, type(ex).__name__)
            trials.sort(key=lambda t: -t[0])
            cv_best, conf = trials[0]
            fit = build(mname, conf, seed).fit(d[FE], d.label)
            auc_tuned = roc_auc_score(e.label, fit.predict_proba(e[FE])[:, 1])
            # default configuration for comparison
            p_def = spatial_cv_pred(models(seed)[mname], d, FE)
            fit_def = models(seed)[mname].fit(d[FE], d.label)
            auc_def = roc_auc_score(e.label, fit_def.predict_proba(e[FE])[:, 1])
            rows.append(dict(seed=seed, train=src, test=tgt, model=mname,
                             cv_AUC_tuned=cv_best, cv_AUC_default=roc_auc_score(d.label, p_def),
                             transfer_AUC_tuned=auc_tuned, transfer_AUC_default=auc_def,
                             cv_AUC_worst_config=min(t[0] for t in trials),
                             transfer_minus_cv_tuned=auc_tuned - cv_best))
            best_confs.append(dict(seed=seed, train=src, model=mname, **{k: str(v) for k, v in conf.items()}))
            print(rows[-1], flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / "tuning_runs.csv", index=False)
pd.DataFrame(best_confs).to_csv(OUT / "tuning_best_configs.csv", index=False)
summ = R.groupby(["train", "model"])[["cv_AUC_default", "cv_AUC_tuned", "transfer_AUC_default", "transfer_AUC_tuned"]].mean().round(3)
summ.to_csv(OUT / "tuning_summary.csv")
print(summ.to_string())
for tr in ("wayanad", "kodagu"):
    s = summ.loc[tr]
    print(f"\n{tr}: CV ranking tuned  ", list(s.cv_AUC_tuned.sort_values(ascending=False).index))
    print(f"{tr}: transfer ranking tuned", list(s.transfer_AUC_tuned.sort_values(ascending=False).index))
