"""R3: inventory comparison and an inventory-completeness (thinning) experiment.
Separates "transfer loss due to inventory differences" from "transfer loss due to environmental shift".
Answers Reviewer 1 point 3 and Reviewer 2 point 4. -> outputs/revision/inventory_*.csv
"""
import json
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from rev_common import *

RAW = ROOT / "data" / "raw"
S = load("no_roads")
FE = feature_sets(S)["S1_all"]
AREA = {"wayanad": 2025.87, "kodagu": 4127.51}          # km2, from the district masks

# ---------------------------------------------------------------- 1. inventory properties
adm = gpd.read_file(RAW / "boundaries" / "geoBoundaries-IND-ADM2.geojson")
ker = gpd.read_file(RAW / "kerala_inventory" / "Kerela landslide.shp").to_crs(CRS := "EPSG:32643")
way_poly = adm[adm.shapeName == "Wayanad"].to_crs(CRS)[["geometry"]]
kod_poly = adm[adm.shapeName == "Kodagu"].to_crs(CRS)[["geometry"]]
way = gpd.sjoin(ker, way_poly, predicate="within")
kod = gpd.read_file(RAW / "kodagu_inventory" / "Landslide_Points.shp").to_crs(CRS)
kodp = gpd.read_file(RAW / "kodagu_inventory" / "Landslide_polygons.shp").to_crs(CRS)

rows = []
for name, g in (("wayanad", way), ("kodagu", kod)):
    xy = np.c_[g.geometry.x.values, g.geometry.y.values]
    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    dist, _ = nn.kneighbors(xy)
    d1 = dist[:, 1]
    lam = len(xy) / (AREA[name] * 1e6)                   # points per m2
    expected_nn = 0.5 / np.sqrt(lam)                     # expected NN distance for a Poisson process
    pos = S[(S.district == name) & (S.label == 1) & (S.seed == 0)]
    rows.append(dict(district=name, n_points=len(g), area_km2=AREA[name],
                     density_per_100km2=round(100 * len(g) / AREA[name], 2),
                     nn_mean_m=round(float(d1.mean()), 1), nn_median_m=round(float(np.median(d1)), 1),
                     clark_evans_R=round(float(d1.mean() / expected_nn), 3),
                     slope_median=round(float(pos.slope.median()), 1),
                     slope_p90=round(float(pos.slope.quantile(0.9)), 1),
                     elev_median=round(float(pos.elevation.median()), 0),
                     tri_median=round(float(pos.tri.median()), 1),
                     rain_aug2018_median=round(float(pos.rain_aug2018.median()), 0)))
INV = pd.DataFrame(rows)
INV.to_csv(OUT / "inventory_properties.csv", index=False)
print(INV.to_string())

# terrain distributions at landslide points: are the two inventories sampling the same terrain?
ks = []
pw = S[(S.district == "wayanad") & (S.label == 1) & (S.seed == 0)]
pk = S[(S.district == "kodagu") & (S.label == 1) & (S.seed == 0)]
for f in CONT:
    st, p = ks_2samp(pw[f], pk[f])
    ks.append(dict(factor=f, KS=round(float(st), 3), p_value=float(p),
                   median_wayanad=round(float(pw[f].median()), 3), median_kodagu=round(float(pk[f].median()), 3)))
KS = pd.DataFrame(ks).sort_values("KS", ascending=False)
KS["p_value_BH"] = fdr_bh(KS.p_value.values)
KS.to_csv(OUT / "inventory_ks_tests.csv", index=False)
print(KS.round(4).to_string())

# landslide area statistics where available
areas = {}
if "Area" in way.columns:
    a = pd.to_numeric(way["Area"], errors="coerce").dropna()
    areas["wayanad_area_m2"] = dict(n=int(len(a)), median=float(a.median()), p90=float(a.quantile(0.9)))
kp = kodp.to_crs(CRS).geometry.area
areas["kodagu_polygon_area_m2"] = dict(n=int(len(kp)), median=float(kp.median()), p90=float(kp.quantile(0.9)))
json.dump(areas, open(OUT / "inventory_areas.json", "w"), indent=1)
print(areas)

# ---------------------------------------------------------------- 2. thinning experiment
# Wayanad has 2.5x the mapped landslide density of Kodagu. If the Kodagu inventory is less complete,
# how much of the transfer loss can incompleteness alone explain? Thin the Wayanad training set and see.
FRACS = [1.0, 0.75, 0.5, 0.41, 0.25]                     # 0.41 matches the Kodagu/Wayanad density ratio
res = []
for seed in SEEDS:
    tr_full = S[(S.district == "wayanad") & (S.seed == seed)].reset_index(drop=True)
    te = S[(S.district == "kodagu") & (S.seed == seed)].reset_index(drop=True)
    pos = tr_full[tr_full.label == 1]; neg = tr_full[tr_full.label == 0]
    blk = pd.Series(blocks(pos), index=pos.index)
    for frac in FRACS:
        for scheme in ("random", "clustered"):
            rng = np.random.default_rng(10_000 + seed)
            if frac == 1.0 and scheme == "clustered":
                continue
            if scheme == "random":
                keep = pos.sample(frac=frac, random_state=seed).index
            else:   # drop whole 5 km blocks: mimics a survey that missed parts of the district
                ub = blk.unique(); rng.shuffle(ub)
                need = int(round(frac * len(pos))); keep, got = [], 0
                for b in ub:
                    idx = blk.index[blk == b]
                    if got >= need:
                        break
                    keep.extend(idx); got += len(idx)
                keep = pd.Index(keep)
            tr = pd.concat([pos.loc[keep], neg.sample(n=len(keep), random_state=seed)]).reset_index(drop=True)
            for mname in ("RF", "XGB"):
                m = models(seed)[mname].fit(tr[FE], tr.label)
                auc_tr = roc_auc_score(te.label, m.predict_proba(te[FE])[:, 1])
                p_cv = spatial_cv_pred(models(seed)[mname], tr, FE)
                res.append(dict(seed=seed, frac=frac, scheme=scheme, model=mname, n_pos=int(len(keep)),
                                transfer_AUC=auc_tr, within_spatial_cv_AUC=roc_auc_score(tr.label, p_cv)))
    print("thinning seed", seed, "done", flush=True)

T = pd.DataFrame(res); T.to_csv(OUT / "inventory_thinning_runs.csv", index=False)
summ = T.groupby(["scheme", "frac", "model"])[["transfer_AUC", "within_spatial_cv_AUC"]].agg(["mean", "std"]).round(3)
summ.to_csv(OUT / "inventory_thinning_summary.csv")
print(summ.to_string())
