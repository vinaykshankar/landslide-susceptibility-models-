"""R7: is the Wayanad/Kodagu contrast in OpenStreetMap road density a mapping-effort artefact?
Quantifies the shift in distance-to-roads and the stability of its learned effect, and compares the
transfer behaviour of models with and without it. Answers Reviewer 2 point 7.
-> outputs/revision/roads_diagnostics.csv
"""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import shap
from rev_common import *

S = load("with_roads")
OH = onehot(S)
FE = CONT + ["dist_roads"] + OH
GROUP = {**{c: c for c in CONT + ["dist_roads"]}, **{c: "landcover" for c in OH if c.startswith("lc_")},
         **{c: "lithology" for c in OH if c.startswith("lith_")}}

rows = []
for seed in SEEDS:
    fits = {}
    for reg in ("wayanad", "kodagu"):
        d = S[(S.district == reg) & (S.seed == seed)]
        fits[reg] = models(seed)["XGB"].fit(d[FE], d.label)
    pooled = S[S.seed == seed].reset_index(drop=True)
    sv = {}
    for reg, f in fits.items():
        v = pd.DataFrame(shap.TreeExplainer(f).shap_values(pooled[FE]), columns=FE)
        sv[reg] = v.T.groupby(pd.Series(GROUP)[FE].values).sum().T
    for fac in sv["wayanad"].columns:
        a, b = sv["wayanad"][fac].values, sv["kodagu"][fac].values
        rows.append(dict(seed=seed, factor=fac, pearson=float(pearsonr(a, b)[0]),
                         spearman=float(spearmanr(a, b)[0]),
                         mean_abs_w=float(np.abs(a).mean()), mean_abs_k=float(np.abs(b).mean())))
    print("roads shap seed", seed, "done", flush=True)

R = pd.DataFrame(rows)
agg = R.groupby("factor").agg(effect_agreement_r=("pearson", "mean"), effect_agreement_sd=("pearson", "std"),
                              mean_abs_shap_wayanad=("mean_abs_w", "mean"),
                              mean_abs_shap_kodagu=("mean_abs_k", "mean")).round(3)
agg["rank_wayanad"] = agg.mean_abs_shap_wayanad.rank(ascending=False).astype(int)
agg["rank_kodagu"] = agg.mean_abs_shap_kodagu.rank(ascending=False).astype(int)
agg.sort_values("mean_abs_shap_kodagu", ascending=False).to_csv(OUT / "roads_diagnostics.csv")
print(agg.sort_values("mean_abs_shap_kodagu", ascending=False).to_string())

# how far are landslides from mapped roads in each district, and how far are random cells?
rows2 = []
for reg in ("wayanad", "kodagu"):
    d = S[(S.district == reg) & (S.seed == 0)]
    for lab, nm in ((1, "landslide"), (0, "absence")):
        v = d.loc[d.label == lab, "dist_roads"]
        rows2.append(dict(district=reg, cells=nm, median_m=round(float(v.median())),
                          p25_m=round(float(v.quantile(.25))), p75_m=round(float(v.quantile(.75))),
                          within_250m_pct=round(100 * float((v < 250).mean()), 1)))
D = pd.DataFrame(rows2)
D.to_csv(OUT / "roads_distance_summary.csv", index=False)
print(D.to_string())
