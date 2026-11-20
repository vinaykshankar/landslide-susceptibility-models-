"""R11: Figure 11, the event-rainfall location-code probes.
(a) within-district spatial CV AUC against transfer AUC for each probe, with the 1:1 line
(b) optimism gap by probe and direction
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rev_common import OUT, ROOT

FIG = ROOT / "outputs" / "no_roads" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})

V = [("V0_all_factors", "V0 all factors", "#52514e"),
     ("V1_no_rainfall", "V1 rainfall removed", "#2a78d6"),
     ("V2_rainfall_replaced_by_spatial_trend", "V2 rainfall → spatial trend", "#9b59c6"),
     ("V3_rainfall_only", "V3 rainfall only", "#eb6834"),
     ("V4_spatial_trend_only", "V4 spatial trend only", "#c0392b"),
     ("V5_terrain_fine_slope_tri", "V5 slope+TRI, 30 m", "#1baf7a"),
     ("V6_terrain_coarsened_to_chirps_grid", "V6 slope+TRI, 0.05° grid", "#eda100")]

R = pd.read_csv(OUT / "rainfall_probe_runs.csv")
cv = R.set_index(["seed", "variant", "model", "train"]).within_spatial_cv_AUC
R["gap"] = [cv.loc[(r.seed, r.variant, r.model, r.test)] - r.transfer_AUC for _, r in R.iterrows()]
g = R.groupby(["variant", "train"]).agg(cv_own=("within_spatial_cv_AUC", "mean"), tr=("transfer_AUC", "mean"),
                                        tr_sd=("transfer_AUC", "std"), gap=("gap", "mean"), gap_sd=("gap", "std"))
# the CV value to pair with a transfer is the TARGET district's own CV
own = R.groupby(["variant", "train"]).within_spatial_cv_AUC.mean()

fig = plt.figure(figsize=(7.2, 3.1))
ax = fig.add_axes([0.065, 0.16, 0.37, 0.80])
ax.plot([0.3, 1.0], [0.3, 1.0], color="#b8b6ae", lw=0.9, ls="-", zorder=1)
ax.text(0.885, 0.872, "1:1", color="#8a8880", fontsize=7, ha="right")
for key, lab, c in V:
    for src, tgt, mk in (("wayanad", "kodagu", "o"), ("kodagu", "wayanad", "^")):
        x = own.loc[(key, tgt)]; y = g.loc[(key, src)].tr
        ax.scatter([x], [y], s=34, marker=mk, color=c, edgecolor="white", linewidth=0.6, zorder=3)
ax.set_xlim(0.72, 1.0); ax.set_ylim(0.26, 0.90)
ax.set_xlabel("Target district's own spatial CV AUC")
ax.set_ylabel("Transfer AUC")
ax.axhline(0.5, color="#52514e", lw=0.6, ls=":")
ax.grid(color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
ax.set_title("(a) Within-district skill against transferred skill", loc="left")

ax2 = fig.add_axes([0.55, 0.16, 0.355, 0.80])
yv = np.arange(len(V))[::-1]
for i, (key, lab, c) in enumerate(V):
    for j, (src, hatch) in enumerate((("wayanad", None), ("kodagu", "////"))):
        r = g.loc[(key, src)]
        ax2.barh(yv[i] + (0.19 if j == 0 else -0.19), r.gap, height=0.34, color=c, alpha=1.0 if j == 0 else 0.45,
                 edgecolor=c, hatch=hatch, linewidth=0.5)
        ax2.errorbar(r.gap, yv[i] + (0.19 if j == 0 else -0.19), xerr=r.gap_sd, fmt="none", ecolor="#0b0b0b",
                     elinewidth=0.6, capsize=1.3)
ax2.axvline(0, color="#0b0b0b", lw=0.8)
ax2.set_yticks(yv); ax2.set_yticklabels([l for _, l, _ in V], fontsize=7)
ax2.set_xlabel("Optimism gap (target CV AUC − transfer AUC)")
ax2.grid(axis="x", color="#e6e5e0", lw=0.5); ax2.set_axisbelow(True)
ax2.set_title("(b) Optimism gap by probe", loc="left")

h = [Line2D([], [], marker="o", ls="none", color="#52514e", ms=5, label="Wayanad → Kodagu"),
     Line2D([], [], marker="^", ls="none", color="#52514e", ms=5, label="Kodagu → Wayanad")]
ax.legend(handles=h, fontsize=7, loc="lower left")
h2 = [Line2D([], [], color="#52514e", lw=6, label="Wayanad → Kodagu"),
      Line2D([], [], color="#52514e", lw=6, alpha=0.45, label="Kodagu → Wayanad")]
ax2.legend(handles=h2, fontsize=7, loc="lower right")
fig.text(0.065, -0.04, "Probe colours are shared between panels; each point is the mean of ten absence samplings and two classifiers.",
         fontsize=7, color="#52514e")
fig.savefig(FIG / "Figure11_rainfall_probes.png")
plt.close(fig)
print("Figure 11 saved")
