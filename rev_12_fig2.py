"""R12: redraw Figure 2 (workflow) so that it includes the diagnostic experiments added in revision."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from rev_common import ROOT

FIG = ROOT / "outputs" / "no_roads" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 70)


def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#52514e", lw=0.6))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2, weight="bold", color="#0b0b0b")
    ax.text(x + w / 2, y + h - 6.3, "\n".join(lines), ha="center", va="top", fontsize=6.8, color="#0b0b0b", linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.8))


light, warm, green, blue = "#eef4fc", "#fdf0ea", "#e8f7f1", "#f3eefc"
box(1, 44, 22, 24, "1 Open data", ["2018 inventories", "(Kerala; Kodagu)", "Copernicus DEM 30 m", "ESA WorldCover 2021",
                                   "CHIRPS rainfall", "SoilGrids 250 m", "GSI geology 1:2M", "OSM roads"], light)
box(27, 44, 22, 24, "2 Factor stack", ["30 m grid, UTM 43N", "16 continuous factors", "(terrain, hydrology,", "rainfall, soil, roads)",
                                       "+ land cover, lithology", "(one-hot)"], light)
box(53, 44, 22, 24, "3 Sampling", ["Landslide cells", "Absence cells > 100 m", "from landslides", "1:1 ratio,", "10 random draws"], light)
box(79, 44, 20, 24, "4 Classifiers", ["Logistic regression", "Random forest", "XGBoost", "LightGBM"], light)

box(1, 16, 30, 23, "5 Validation designs", ["Random 10-fold CV", "Spatial 5-fold CV (5 km blocks)", "Transfer Wayanad → Kodagu",
                                            "Transfer Kodagu → Wayanad", "AUC, accuracy, F1, kappa, Brier"], warm)
box(35, 16, 30, 23, "6 Shift diagnostics", ["Background samples (20,000 cells", "per district)", "Standardised mean difference,",
                                            "Wasserstein distance,", "adversarial AUC, decomposed"], green)
box(69, 16, 30, 23, "7 Explain & adapt", ["TreeSHAP (XGBoost)", "Rank agreement and effect", "agreement across districts",
                                          "S2: regional z-scores; S3, S4:", "drop shifted factors; maps"], green)

box(1, 1, 98, 12, "8 Diagnostic experiments", [
    "Factor subsets passing both diagnostics  ·  Inventory thinning  ·  Hyperparameter tuning",
    "Evaluation restricted to steep terrain  ·  Location-code probes for event rainfall"], blue)

arrow(23.8, 56, 26.2, 56); arrow(49.8, 56, 52.2, 56); arrow(75.8, 56, 78.2, 56)
arrow(89, 43.2, 89, 41); arrow(89, 41, 16, 41); arrow(16, 41, 16, 39.8)
arrow(31.8, 27.5, 34.2, 27.5); arrow(65.8, 27.5, 68.2, 27.5)
arrow(84, 15.2, 84, 13.4); arrow(16, 15.2, 16, 13.4); arrow(50, 15.2, 50, 13.4)

fig.savefig(FIG / "Figure2_workflow.png")
plt.close(fig)
print("Figure 2 redrawn")
