"""R6: is the event-rainfall factor acting as a spatial location code?
(a) replace rainfall by a smooth spatial trend; (b) coarsen terrain factors to the CHIRPS grid;
(c) rainfall-only models. Tests the interpretation in Discussion 5.2 instead of asserting it.
Answers Reviewer 1 point 8 and Reviewer 2 point 8. -> outputs/revision/rainfall_*.csv
"""
import numpy as np, pandas as pd, rasterio
from sklearn.metrics import roc_auc_score
from rev_common import *

S = load("no_roads").copy()
SETS = feature_sets(S)
BASE = SETS["S1_all"]
NORAIN = SETS["S4_no_rainfall"]
RAIN = ["rain_annual_mean", "rain_aug2018"]
COARSE_PX = 183                      # 183 x 30 m = 5.49 km, the CHIRPS 0.05 deg cell size

# ---- smooth spatial trend, standardised within district so the model cannot read absolute position
for reg in ("wayanad", "kodagu"):
    m = S.district == reg
    xs = (S.loc[m, "x"] - S.loc[m, "x"].mean()) / S.loc[m, "x"].std()
    ys = (S.loc[m, "y"] - S.loc[m, "y"].mean()) / S.loc[m, "y"].std()
    S.loc[m, "sx"], S.loc[m, "sy"] = xs, ys
    S.loc[m, "sx2"], S.loc[m, "sy2"], S.loc[m, "sxy"] = xs ** 2, ys ** 2, xs * ys
TREND = ["sx", "sy", "sx2", "sy2", "sxy"]

# ---- terrain factors block-averaged to the CHIRPS grid (same support as the rainfall fields)
for f in ("slope", "tri"):
    for reg in ("wayanad", "kodagu"):
        with rasterio.open(PROC / reg / f"{f}.tif") as s:
            a = s.read(1).astype("float32"); T = s.transform
        H, W = a.shape
        ph, pw = int(np.ceil(H / COARSE_PX)), int(np.ceil(W / COARSE_PX))
        pad = np.full((ph * COARSE_PX, pw * COARSE_PX), np.nan, dtype="float32"); pad[:H, :W] = a
        blk = np.nanmean(pad.reshape(ph, COARSE_PX, pw, COARSE_PX), axis=(1, 3))
        m = S.district == reg
        cols = ((S.loc[m, "x"].values - T.c) / T.a / COARSE_PX).astype(int)
        rows = ((S.loc[m, "y"].values - T.f) / T.e / COARSE_PX).astype(int)
        S.loc[m, f + "_coarse"] = blk[np.clip(rows, 0, ph - 1), np.clip(cols, 0, pw - 1)]
COARSE = ["slope_coarse", "tri_coarse"]

VARIANTS = {
    "V0_all_factors": BASE,
    "V1_no_rainfall": NORAIN,
    "V2_rainfall_replaced_by_spatial_trend": NORAIN + TREND,
    "V3_rainfall_only": RAIN,
    "V4_spatial_trend_only": TREND,
    "V5_terrain_fine_slope_tri": ["slope", "tri"],
    "V6_terrain_coarsened_to_chirps_grid": COARSE,
}

rows = []
for seed in SEEDS:
    for name, feats in VARIANTS.items():
        for src, tgt in (("wayanad", "kodagu"), ("kodagu", "wayanad")):
            d = S[(S.district == src) & (S.seed == seed)].reset_index(drop=True)
            e = S[(S.district == tgt) & (S.seed == seed)].reset_index(drop=True)
            d = d.dropna(subset=feats); e = e.dropna(subset=feats)
            for mname in ("RF", "XGB"):
                p_cv = spatial_cv_pred(models(seed)[mname], d, feats)
                fit = models(seed)[mname].fit(d[feats], d.label)
                rows.append(dict(seed=seed, variant=name, model=mname, train=src, test=tgt,
                                 within_spatial_cv_AUC=roc_auc_score(d.label, p_cv),
                                 transfer_AUC=roc_auc_score(e.label, fit.predict_proba(e[feats])[:, 1])))
    print("rainfall-probe seed", seed, "done", flush=True)

R = pd.DataFrame(rows); R.to_csv(OUT / "rainfall_probe_runs.csv", index=False)
summ = R.groupby(["variant", "train", "model"])[["within_spatial_cv_AUC", "transfer_AUC"]].agg(["mean", "std"]).round(3)
summ.to_csv(OUT / "rainfall_probe_summary.csv")
print(summ.to_string())
