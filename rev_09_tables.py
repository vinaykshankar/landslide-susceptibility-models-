"""R9: build the new manuscript tables (and supplementary tables) from the revision outputs.
Extends manuscript/tables.json in place. -> tables table8..table13, tableS2..tableS5
"""
import json
import numpy as np, pandas as pd
from rev_common import OUT, ROOT, MODELS

M = ROOT / "manuscript"
T = json.load(open(M / "tables.json"))
N = json.load(open(M / "numbers.json"))
LBL = {"S1_all": "S1 all factors", "S3_no_event_rain": "S3 event rainfall removed", "S4_no_rainfall": "S4 all rainfall removed",
       "T1_terrain_only": "T1 terrain and hydrology only", "T2_terrain_elev": "T2 terrain, hydrology and elevation",
       "T3_stable_only": "T3 unshifted factors only"}

# ---------------- Table 8: adversarial decomposition
A = pd.read_csv(OUT / "adversarial_decomposition.csv")
rows = [["Factor set", "Factors", "Largest single-factor AUC", "Multivariate adversarial AUC"]]
for _, r in A.iterrows():
    rows.append([r["set"][0].upper() + r["set"][1:], str(int(r.n_feat)),
                 "–" if np.isnan(r.max_marginal_AUC) else f"{r.max_marginal_AUC:.2f}", f"{r.adv_AUC:.3f}"])
T["table8"] = rows
G = pd.read_csv(OUT / "adversarial_greedy.csv")
N["adv_greedy"] = {int(r.step): dict(added=r.added, auc=round(float(r.cumulative_adv_AUC), 3)) for _, r in G.iterrows()}
N["adv_decomp"] = {r["set"]: round(float(r.adv_AUC), 3) for _, r in A.iterrows()}

# ---------------- Table 9: factor subsets, transfer AUC and optimism gap
F = pd.read_csv(OUT / "factorsets_runs.csv")
cv = F[F.scheme == "spatial_cv"].set_index(["seed", "fset", "model", "train"]).AUC
tr = F[F.scheme == "transfer"].copy()
tr["own_cv"] = [cv.loc[(r.seed, r.fset, r.model, r.test)] for _, r in tr.iterrows()]
tr["gap"] = tr.own_cv - tr.AUC
summ = tr.groupby(["train", "fset", "model"]).agg(AUC=("AUC", "mean"), SD=("AUC", "std"), gap=("gap", "mean")).reset_index()
cvm = F[F.scheme == "spatial_cv"].groupby(["train", "fset", "model"]).AUC.mean()
rows = [["Direction", "Factor set", "Model", "Within-district CV (source)", "Transfer AUC", "Optimism gap"]]
for src, tgt in (("wayanad", "kodagu"), ("kodagu", "wayanad")):
    for fs in ["S1_all", "S4_no_rainfall", "T1_terrain_only", "T2_terrain_elev", "T3_stable_only"]:
        for m in MODELS:
            x = summ[(summ.train == src) & (summ.fset == fs) & (summ.model == m)].iloc[0]
            rows.append([f"{src.capitalize()} → {tgt.capitalize()}", LBL[fs], m,
                         f"{cvm.loc[(src, fs, m)]:.3f}", f"{x.AUC:.3f} ± {x.SD:.3f}", f"{x.gap:+.3f}"])
T["table9"] = rows
N["factorsets"] = {f"{r.train}|{r.fset}|{r.model}": dict(auc=round(float(r.AUC), 3), sd=round(float(r.SD), 3),
                                                          gap=round(float(r.gap), 3)) for _, r in summ.iterrows()}
N["factorsets_cv"] = {f"{a}|{b}|{c}": round(float(v), 3) for (a, b, c), v in cvm.items()}

# ---------------- Table 10: steep-terrain evaluation
S = pd.read_csv(OUT / "steep_eval_summary.csv")
cols = ["AUC_slope_ge_0", "AUC_slope_ge_15", "AUC_slope_ge_20", "AUC_slope_ge_25"]
rows = [["Factor set", "Design", "Trained in", "Model", "All cells", "Slope ≥ 15°", "Slope ≥ 20°", "Slope ≥ 25°"]]
for _, r in S.iterrows():
    if r.model not in ("RF", "XGB"):
        continue
    rows.append([LBL.get(r.fset, r.fset), "Spatial CV" if r.scheme == "spatial_cv" else "Transfer",
                 r.train.capitalize(), r.model] + [f"{r[c]:.3f}" for c in cols])
T["table10"] = rows
N["steep"] = {f"{r.fset}|{r.scheme}|{r.train}|{r.model}": {c.replace('AUC_slope_ge_', 'ge'): round(float(r[c]), 3) for c in cols}
              for _, r in S.iterrows()}

# ---------------- Table 11: inventory properties and thinning
I = pd.read_csv(OUT / "inventory_properties.csv")
rows = [["Property", "Wayanad", "Kodagu"]]
pretty = [("n_points", "Mapped landslides in district", "{:.0f}"), ("area_km2", "District area (km²)", "{:,.0f}"),
          ("density_per_100km2", "Mapped density (per 100 km²)", "{:.1f}"),
          ("nn_median_m", "Median nearest-neighbour distance (m)", "{:.0f}"),
          ("clark_evans_R", "Clark–Evans aggregation index", "{:.2f}"),
          ("slope_median", "Median slope at landslide cells (°)", "{:.1f}"),
          ("tri_median", "Median TRI at landslide cells (m)", "{:.1f}"),
          ("elev_median", "Median elevation at landslide cells (m)", "{:,.0f}"),
          ("rain_aug2018_median", "Median August 2018 rainfall at landslide cells (mm)", "{:,.0f}")]
for k, lab, fmt in pretty:
    rows.append([lab, fmt.format(I.loc[I.district == "wayanad", k].iloc[0]), fmt.format(I.loc[I.district == "kodagu", k].iloc[0])])
T["table11"] = rows
N["inventory"] = {r.district: r.to_dict() for _, r in I.iterrows()}

TH = pd.read_csv(OUT / "inventory_thinning_runs.csv")
g = TH.groupby(["scheme", "frac", "model"])[["transfer_AUC", "within_spatial_cv_AUC"]].agg(["mean", "std"])
rows = [["Thinning", "Landslide cells kept", "Model", "Within-district CV", "Transfer AUC (→ Kodagu)"]]
for (sch, frac, m), r in g.iterrows():
    if m not in ("RF", "XGB"):
        continue
    rows.append([{"random": "Random", "clustered": "Whole 5 km blocks"}[sch], f"{frac:.0%}", m,
                 f"{r[('within_spatial_cv_AUC', 'mean')]:.3f} ± {r[('within_spatial_cv_AUC', 'std')]:.3f}",
                 f"{r[('transfer_AUC', 'mean')]:.3f} ± {r[('transfer_AUC', 'std')]:.3f}"])
T["table12"] = rows
N["thinning"] = {f"{s}|{f}|{m}": dict(transfer=round(float(v[("transfer_AUC", "mean")]), 3),
                                      cv=round(float(v[("within_spatial_cv_AUC", "mean")]), 3))
                 for (s, f, m), v in g.iterrows()}

# ---------------- Supplementary: FDR table and SHAP metric robustness
W = pd.read_csv(OUT / "wilcoxon_fdr.csv")
rows = [["Family", "Comparison", "Mean difference", "Samplings favouring", "p (raw)", "p (BH-adjusted)"]]
PRETTY = {"->": " \u2192 ", "_": " ", "wayanad": "Wayanad", "kodagu": "Kodagu", "S2 regional z": "S2 regional standardisation",
          "S3 drop shifted": "S3 shifted factors dropped", "S4 no rainfall": "S4 rainfall removed",
          "T1 terrain only": "T1 terrain only", "T2 terrain elev": "T2 terrain + elevation", "T3 stable only": "T3 unshifted only"}
def pretty(c):
    for a, b in PRETTY.items():
        c = c.replace(a, b)
    return c
for _, r in W.iterrows():
    rows.append([r.family.capitalize(), pretty(r.comparison), f"{r.mean_diff:+.3f}", f"{int(r.wins)}/{int(r.n)}",
                 f"{r.p_raw:.4f}", f"{r.p_BH:.4f}"])
T["tableS2"] = rows
N["fdr"] = dict(n_tests=int(len(W)), sig_raw=int(W["sig_raw_0.05"].sum()), sig_bh=int(W["sig_BH_0.05"].sum()))

R = pd.read_csv(OUT / "shap_metric_robustness.csv")
main = R[(R.algo == "tree_path_dependent") & (R.eval_sample == "pooled")].sort_values("pearson_mean", ascending=False)
rows = [["Factor", "Pearson r (95% CI)", "Spearman ρ", "Lin's CCC", "Pearson r, interventional SHAP", "Pearson r, Kodagu samples only"]]
piv = R.pivot_table(index="factor", columns=["algo", "eval_sample"], values="pearson_mean")
for _, r in main.iterrows():
    rows.append([r.factor, f"{r.pearson_mean:+.2f} ({r.pearson_lo95:+.2f}, {r.pearson_hi95:+.2f})",
                 f"{r.spearman_mean:+.2f}", f"{r.ccc_mean:+.2f}",
                 f"{piv.loc[r.factor, ('interventional', 'pooled')]:+.2f}",
                 f"{piv.loc[r.factor, ('tree_path_dependent', 'kodagu_samples')]:+.2f}"])
T["tableS3"] = rows
N["shap_robust"] = {r.factor: dict(pearson=round(float(r.pearson_mean), 3), lo=round(float(r.pearson_lo95), 3),
                                   hi=round(float(r.pearson_hi95), 3), spearman=round(float(r.spearman_mean), 3),
                                   ccc=round(float(r.ccc_mean), 3)) for _, r in main.iterrows()}

json.dump(T, open(M / "tables.json", "w"), indent=1, ensure_ascii=False)
json.dump(N, open(M / "numbers.json", "w"), indent=1, ensure_ascii=False)
print("tables now:", sorted(T))
# ---------------- Supplementary Table S5: hyperparameter tuning
if (OUT / "tuning_runs.csv").exists():
    TU = pd.read_csv(OUT / "tuning_runs.csv")
    g = TU.groupby(["train", "model"])[["cv_AUC_default", "cv_AUC_tuned", "transfer_AUC_default", "transfer_AUC_tuned"]].mean()
    rows = [["Trained in", "Model", "Spatial CV, default", "Spatial CV, tuned", "Transfer, default", "Transfer, tuned"]]
    for (tr, m), r in g.iterrows():
        rows.append([tr.capitalize(), m, f"{r.cv_AUC_default:.3f}", f"{r.cv_AUC_tuned:.3f}",
                     f"{r.transfer_AUC_default:.3f}", f"{r.transfer_AUC_tuned:.3f}"])
    T["tableS4"] = rows
    N["tuning"] = {f"{a}|{b}": {k: round(float(v), 3) for k, v in r.items()} for (a, b), r in g.iterrows()}
    N["tuning_rank"] = {tr: dict(cv=list(g.loc[tr].cv_AUC_tuned.sort_values(ascending=False).index),
                                 transfer=list(g.loc[tr].transfer_AUC_tuned.sort_values(ascending=False).index))
                        for tr in ("wayanad", "kodagu")}

# ---------------- Table 13: rainfall mechanism probes
if (OUT / "rainfall_probe_runs.csv").exists():
    RP = pd.read_csv(OUT / "rainfall_probe_runs.csv")
    g = RP.groupby(["variant", "train", "model"])[["within_spatial_cv_AUC", "transfer_AUC"]].mean()
    names = {"V0_all_factors": "All factors", "V1_no_rainfall": "Rainfall removed",
             "V2_rainfall_replaced_by_spatial_trend": "Rainfall replaced by smooth spatial trend",
             "V3_rainfall_only": "Rainfall only", "V4_spatial_trend_only": "Spatial trend only",
             "V5_terrain_fine_slope_tri": "Slope and TRI at 30 m", "V6_terrain_coarsened_to_chirps_grid": "Slope and TRI at the 0.05° rainfall grid"}
    rows = [["Factor set", "Model", "Within-district CV (Wayanad)", "Transfer → Kodagu", "Within-district CV (Kodagu)", "Transfer → Wayanad"]]
    for v, lab in names.items():
        for m in ("RF", "XGB"):
            try:
                w = g.loc[(v, "wayanad", m)]; k = g.loc[(v, "kodagu", m)]
            except KeyError:
                continue
            rows.append([lab, m, f"{w.within_spatial_cv_AUC:.3f}", f"{w.transfer_AUC:.3f}",
                         f"{k.within_spatial_cv_AUC:.3f}", f"{k.transfer_AUC:.3f}"])
    T["table13"] = rows
    N["rainfall_probe"] = {f"{a}|{b}|{c}": dict(cv=round(float(r.within_spatial_cv_AUC), 3), transfer=round(float(r.transfer_AUC), 3))
                           for (a, b, c), r in g.iterrows()}

# ---------------- Table 14 / S6: roads diagnostics
if (OUT / "roads_diagnostics.csv").exists():
    RD = pd.read_csv(OUT / "roads_diagnostics.csv", index_col=0)
    DS = pd.read_csv(OUT / "roads_distance_summary.csv")
    rows = [["Measure", "Wayanad", "Kodagu"]]
    rows.append(["Mapped road density (km km⁻²)", "2.89", "0.74"])
    for cells in ("landslide", "absence"):
        w = DS[(DS.district == "wayanad") & (DS.cells == cells)].iloc[0]
        k = DS[(DS.district == "kodagu") & (DS.cells == cells)].iloc[0]
        rows.append([f"Median distance to a mapped road, {cells} cells (m)", f"{w.median_m:,.0f}", f"{k.median_m:,.0f}"])
    if "dist_roads" in RD.index:
        rows.append(["Cells within 250 m of a mapped road, landslide cells (%)",
                     f"{DS[(DS.district=='wayanad')&(DS.cells=='landslide')].iloc[0].within_250m_pct:.1f}",
                     f"{DS[(DS.district=='kodagu')&(DS.cells=='landslide')].iloc[0].within_250m_pct:.1f}"])
        rows.append(["Cells within 250 m of a mapped road, absence cells (%)",
                     f"{DS[(DS.district=='wayanad')&(DS.cells=='absence')].iloc[0].within_250m_pct:.1f}",
                     f"{DS[(DS.district=='kodagu')&(DS.cells=='absence')].iloc[0].within_250m_pct:.1f}"])
        rows.append(["SHAP importance rank of distance to roads", str(int(RD.loc["dist_roads", "rank_wayanad"])),
                     str(int(RD.loc["dist_roads", "rank_kodagu"]))])
        rows.append(["SHAP effect agreement between districts (r)",
                     f"{RD.loc['dist_roads', 'effect_agreement_r']:+.2f} ± {RD.loc['dist_roads', 'effect_agreement_sd']:.2f}", ""])
    T["tableS5"] = rows
    N["roads"] = RD.loc["dist_roads"].to_dict() if "dist_roads" in RD.index else {}

json.dump(T, open(M / "tables.json", "w"), indent=1, ensure_ascii=False)
json.dump(N, open(M / "numbers.json", "w"), indent=1, ensure_ascii=False)
print("final tables:", sorted(T))

for k in ("table8", "table9", "table10", "table11", "table12"):
    print(k, len(T[k]) - 1, "rows")
