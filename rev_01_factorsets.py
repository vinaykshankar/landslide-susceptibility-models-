"""R1: factor-set experiments (terrain-only transfer, stable-factor transfer) + steep-terrain evaluation.
Answers Reviewer 2 points 1 and 5. Writes outputs/revision/factorsets_runs.csv and steep_eval_runs.csv.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from rev_common import *

S = load("no_roads")
SETS = feature_sets(S)
rows, steep = [], []

for seed in SEEDS:
    for name, feats in SETS.items():
        for reg in ("wayanad", "kodagu"):
            d = S[(S.district == reg) & (S.seed == seed)].reset_index(drop=True)
            tgt = "kodagu" if reg == "wayanad" else "wayanad"
            e = S[(S.district == tgt) & (S.seed == seed)].reset_index(drop=True)
            for mname, m in models(seed).items():
                # within-district spatial block CV
                p_cv = spatial_cv_pred(m, d, feats)
                rows.append(dict(seed=seed, fset=name, scheme="spatial_cv", train=reg, test=reg, model=mname,
                                 AUC=roc_auc_score(d.label, p_cv), n_feat=len(feats)))
                steep.append(dict(seed=seed, fset=name, scheme="spatial_cv", train=reg, test=reg, model=mname,
                                  **auc_subsets(d, p_cv)))
                # cross-district transfer
                fit = models(seed)[mname].fit(d[feats], d.label)
                p_tr = fit.predict_proba(e[feats])[:, 1]
                rows.append(dict(seed=seed, fset=name, scheme="transfer", train=reg, test=tgt, model=mname,
                                 AUC=roc_auc_score(e.label, p_tr), n_feat=len(feats)))
                steep.append(dict(seed=seed, fset=name, scheme="transfer", train=reg, test=tgt, model=mname,
                                  **auc_subsets(e, p_tr)))
    print("seed", seed, "done", flush=True)

R = pd.DataFrame(rows); R.to_csv(OUT / "factorsets_runs.csv", index=False)
pd.DataFrame(steep).to_csv(OUT / "steep_eval_runs.csv", index=False)
summ = R.groupby(["scheme", "train", "test", "fset", "model"]).AUC.agg(["mean", "std"]).round(3).reset_index()
summ.to_csv(OUT / "factorsets_summary.csv", index=False)
print(summ.to_string())
