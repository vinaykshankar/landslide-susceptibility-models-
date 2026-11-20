"""R10: new figures for the revision.
Figure 9  transfer AUC and optimism gap by factor set
Figure 10 AUC against the minimum slope of the evaluation subset
Figure 11 mechanism probes for event rainfall (written by rev_11 once rev_06 has run)
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rev_common import OUT, ROOT

FIG = ROOT / "outputs" / "no_roads" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
C = {"LR": "#2a78d6", "RF": "#eb6834", "XGB": "#1baf7a", "LGBM": "#eda100"}
MODELS = ["LR", "RF", "XGB", "LGBM"]
SETS = [("S1_all", "S1\nall factors"), ("S4_no_rainfall", "S4\nno rainfall"),
        ("T1_terrain_only", "T1\nterrain"), ("T2_terrain_elev", "T2\nterrain+elev"), ("T3_stable_only", "T3\nunshifted")]

F = pd.read_csv(OUT / "factorsets_runs.csv")
cv = F[F.scheme == "spatial_cv"].set_index(["seed", "fset", "model", "train"]).AUC
tr = F[F.scheme == "transfer"].copy()
tr["gap"] = [cv.loc[(r.seed, r.fset, r.model, r.test)] - r.AUC for _, r in tr.iterrows()]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
for col, (src, tgt) in enumerate([("wayanad", "kodagu"), ("kodagu", "wayanad")]):
    for row, (val, lab, lo, hi) in enumerate([("AUC", "Transfer AUC", 0.6, 0.9), ("gap", "Optimism gap", -0.06, 0.24)]):
        ax = axes[row, col]
        xw = 0.19
        for j, m in enumerate(MODELS):
            mu = [tr[(tr.train == src) & (tr.fset == f) & (tr.model == m)][val].mean() for f, _ in SETS]
            sd = [tr[(tr.train == src) & (tr.fset == f) & (tr.model == m)][val].std() for f, _ in SETS]
            x = np.arange(len(SETS)) + (j - 1.5) * xw
            if row == 0:
                ax.bar(x, np.array(mu) - lo, bottom=lo, width=xw - 0.03, color=C[m], label=m)
            else:
                ax.bar(x, mu, width=xw - 0.03, color=C[m], label=m)
            ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor="#0b0b0b", elinewidth=0.7, capsize=1.5)
        ax.set_ylim(lo, hi)
        if row == 1:
            ax.axhline(0, color="#0b0b0b", lw=0.8)
            ax.set_xticks(range(len(SETS))); ax.set_xticklabels([l for _, l in SETS], fontsize=7)
        ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel(lab)
        if row == 0:
            ax.set_title(f"({'ab'[col]}) {src.capitalize()} → {tgt.capitalize()}", loc="left")
fig.legend(handles=[Patch(color=C[m], label=m) for m in MODELS], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
fig.text(0.5, -0.01, "Optimism gap = target district's own spatial cross-validation AUC − transfer AUC", ha="center", fontsize=7.5, color="#52514e")
fig.tight_layout(); fig.savefig(FIG / "Figure9_factor_subsets.png"); plt.close(fig)
print("Figure 9 saved")

# ---------------- Figure 10: AUC vs slope threshold
S = pd.read_csv(OUT / "steep_eval_runs.csv")
cuts = [0, 10, 15, 20, 25]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, (src, tgt) in zip(axes, [("wayanad", "kodagu"), ("kodagu", "wayanad")]):
    for fset, ls, lab in (("S1_all", "-", "S1 all factors"), ("T2_terrain_elev", "--", "T2 terrain+elevation")):
        for m in ("RF", "XGB"):
            d = S[(S.fset == fset) & (S.scheme == "transfer") & (S.train == src) & (S.model == m)]
            y = [d[f"AUC_slope_ge_{c}"].mean() for c in cuts]
            ax.plot(cuts, y, ls, color=C[m], marker="o", ms=3.5, lw=1.6, label=f"{m}, {lab}")
    ax.axhline(0.5, color="#52514e", lw=0.6, ls=":")
    ax.set_xlabel("Minimum slope of evaluation cells (°)")
    ax.set_title(f"{src.capitalize()} → {tgt.capitalize()}", loc="left")
    ax.grid(color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("Transfer AUC")
axes[1].legend(fontsize=6.8, loc="upper right", ncol=1)
fig.tight_layout(); fig.savefig(FIG / "Figure10_steep_terrain.png"); plt.close(fig)
print("Figure 10 saved")
