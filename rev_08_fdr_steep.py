"""R8: Benjamini-Hochberg correction applied to every paired Wilcoxon test in the paper,
plus the steep-terrain evaluation summary. Answers Reviewer 1 point 7 and Reviewer 2 point 5.
-> outputs/revision/wilcoxon_fdr.csv, steep_eval_summary.csv
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from rev_common import OUT, ROOT, fdr_bh, MODELS

NR = ROOT / "outputs" / "no_roads" / "tables"

# ---------------- 1. all paired tests, raw and BH-adjusted
P = pd.read_csv(NR / "performance_all_runs.csv")
cv4 = pd.read_csv(NR / "spatial_cv_no_rainfall_runs.csv") if (NR / "spatial_cv_no_rainfall_runs.csv").exists() else None
F = pd.read_csv(OUT / "factorsets_runs.csv")
rows = []


def add(a, b, label, family):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) != len(b) or len(a) < 5:
        return
    stat, p = wilcoxon(a, b)
    rows.append(dict(family=family, comparison=label, mean_diff=float(np.mean(b - a)),
                     wins=int(np.sum(b > a)), n=len(a), p_raw=float(p)))


for m in MODELS:
    for reg in ("wayanad", "kodagu"):
        r = P[(P.scheme == "random_10fold") & (P.train == reg) & (P.model == m)].sort_values("seed").AUC
        sp = P[(P.scheme == "spatial_block5km") & (P.train == reg) & (P.model == m)].sort_values("seed").AUC
        add(r, sp, f"{reg}: spatial CV - random CV | {m}", "validation design")
    for src, tgt in (("wayanad", "kodagu"), ("kodagu", "wayanad")):
        s1 = P[(P.scheme == "transfer") & (P.train == src) & (P.strategy == "S1_all") & (P.model == m)].sort_values("seed").AUC
        loc = P[(P.scheme == "spatial_block5km") & (P.train == tgt) & (P.model == m)].sort_values("seed").AUC
        add(loc, s1, f"{src}->{tgt}: transfer - target spatial CV | {m}", "validation design")
        for st in ("S2_regional_z", "S3_drop_shifted", "S4_no_rainfall"):
            x = P[(P.scheme == "transfer") & (P.train == src) & (P.strategy == st) & (P.model == m)].sort_values("seed").AUC
            add(s1, x, f"{src}->{tgt}: {st} - S1 | {m}", "adaptation strategy")
        # new factor subsets vs S1
        base = F[(F.scheme == "transfer") & (F.train == src) & (F.fset == "S1_all") & (F.model == m)].sort_values("seed").AUC
        for fs in ("T1_terrain_only", "T2_terrain_elev", "T3_stable_only"):
            x = F[(F.scheme == "transfer") & (F.train == src) & (F.fset == fs) & (F.model == m)].sort_values("seed").AUC
            add(base, x, f"{src}->{tgt}: {fs} - S1 (transfer) | {m}", "factor subset")
        # optimism gap vs zero is reported descriptively, not tested
W = pd.DataFrame(rows)
W["p_BH"] = fdr_bh(W.p_raw.values)
W["sig_raw_0.05"] = W.p_raw < 0.05
W["sig_BH_0.05"] = W.p_BH < 0.05
W = W.round(4)
W.to_csv(OUT / "wilcoxon_fdr.csv", index=False)
print(W.groupby("family")[["sig_raw_0.05", "sig_BH_0.05"]].sum().to_string())
print("tests:", len(W), "| significant raw:", int(W["sig_raw_0.05"].sum()), "| significant after BH:", int(W["sig_BH_0.05"].sum()))
flipped = W[W["sig_raw_0.05"] & ~W["sig_BH_0.05"]]
print("\nlose significance after BH:", len(flipped))
print(flipped[["comparison", "mean_diff", "p_raw", "p_BH"]].to_string(index=False) if len(flipped) else "none")

# ---------------- 2. steep-terrain evaluation
S = pd.read_csv(OUT / "steep_eval_runs.csv")
cols = [c for c in S.columns if c.startswith("AUC_slope_ge_")]
g = S[S.fset.isin(["S1_all", "T2_terrain_elev"])].groupby(["fset", "scheme", "train", "model"])[cols].mean().round(3)
g.to_csv(OUT / "steep_eval_summary.csv")
print("\nAUC by minimum slope of the evaluation subset:")
print(g.to_string())
n = S[(S.fset == "S1_all") & (S.scheme == "transfer")].groupby("test")[[c for c in S.columns if c.startswith(("n_slope", "pos_frac"))]].mean().round(2)
print("\nsubset sizes and landslide fraction:")
print(n.to_string())
