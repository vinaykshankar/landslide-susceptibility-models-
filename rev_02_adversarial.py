"""R2: why is the multivariate adversarial AUC 1.00 when no single factor exceeds 0.81?
Decomposes domain separability by factor group, by marginal-shift stratum, and with SHAP.
Answers Reviewer 2 point 2. -> outputs/revision/adversarial_decomposition.csv, adversarial_shap.csv
"""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap
from rev_common import OUT, PROC, CONT, TERRAIN, STABLE_CONT, onehot

BG = pd.read_csv(PROC / "background_no_roads.csv")
OH = onehot(BG)
y = (BG.district == "kodagu").astype(int).values
shift = pd.read_csv(OUT.parent / "no_roads" / "tables" / "covariate_shift.csv").set_index("feature")


def adv(features, label, reps=3):
    """Cross-validated adversarial AUC (district classifier) for a feature subset."""
    aucs = []
    for r in range(reps):
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=2, random_state=r, eval_metric="logloss")
        p = cross_val_predict(m, BG[features], y, cv=StratifiedKFold(5, shuffle=True, random_state=r),
                              method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    return dict(set=label, n_feat=len(features), adv_AUC=float(np.mean(aucs)), adv_AUC_sd=float(np.std(aucs)),
                max_marginal_AUC=float(max(shift.adv_AUC.get(f, np.nan) for f in features if f in shift.index)
                                       if any(f in shift.index for f in features) else np.nan))


rows = []
RAIN = ["rain_annual_mean", "rain_aug2018"]
SOIL = ["soil_clay", "soil_sand", "soil_bdod"]
HYD = ["twi", "spi", "dist_streams"]
TER = ["slope", "tri", "plan_curvature", "profile_curvature", "northness", "eastness"]
rows.append(adv(CONT + OH, "all factors"))
rows.append(adv(CONT, "all continuous"))
rows.append(adv([f for f in CONT if f not in RAIN] + OH, "all minus rainfall"))
rows.append(adv([f for f in CONT if f not in RAIN + ["elevation"]] + OH, "all minus rainfall and elevation"))
rows.append(adv(STABLE_CONT + OH, "stable factors only (marginal AUC <= 0.55)"))
rows.append(adv(TERRAIN, "terrain and hydrology only"))
rows.append(adv(TER, "terrain derivatives only"))
rows.append(adv(HYD, "hydrology only"))
rows.append(adv(SOIL, "soil only"))
rows.append(adv(RAIN, "rainfall only"))
rows.append(adv(["elevation"], "elevation only"))
rows.append(adv(OH, "lithology and land cover only"))
rows.append(adv(["lith_" + c for c in ["charnockite", "khondalite_migmatite", "peninsular_gneiss", "schist_belt", "intrusive"]], "lithology only"))

# greedy forward selection: how many factors are needed to separate the districts?
pool = CONT + OH
chosen, greedy = [], []
cur = []
for step in range(6):
    best = None
    for f in pool:
        if f in chosen:
            continue
        r = adv(chosen + [f], "tmp", reps=1)
        if best is None or r["adv_AUC"] > best[1]:
            best = (f, r["adv_AUC"])
    chosen.append(best[0])
    greedy.append(dict(step=step + 1, added=best[0], cumulative_adv_AUC=round(best[1], 4),
                       marginal_AUC=float(shift.adv_AUC.get(best[0], np.nan))))
    print("greedy", greedy[-1], flush=True)
pd.DataFrame(greedy).to_csv(OUT / "adversarial_greedy.csv", index=False)

D = pd.DataFrame(rows)
D.to_csv(OUT / "adversarial_decomposition.csv", index=False)
print(D.round(3).to_string())

# SHAP attribution of the full adversarial model
m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  n_jobs=2, random_state=0, eval_metric="logloss").fit(BG[CONT + OH], y)
sv = pd.DataFrame(shap.TreeExplainer(m).shap_values(BG[CONT + OH]), columns=CONT + OH)
grp = {**{c: c for c in CONT}, **{c: "land cover" for c in OH if c.startswith("lc_")},
       **{c: "lithology" for c in OH if c.startswith("lith_")}}
imp = sv.abs().mean().groupby(pd.Series(grp)).sum().sort_values(ascending=False)
imp.name = "mean_abs_shap_adversarial"
out = imp.to_frame()
out["marginal_adv_AUC"] = [shift.adv_AUC.get(i, np.nan) for i in out.index]
out.to_csv(OUT / "adversarial_shap.csv")
print(out.round(3).to_string())
json.dump({"note": "adversarial validation of district membership on 20,000 background cells per district"},
          open(OUT / "adversarial_meta.json", "w"))
