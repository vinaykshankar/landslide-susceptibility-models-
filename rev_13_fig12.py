"""R13: Figure 12 - the two stages of the proposed workflow, drawn as a flowchart.
Stage A (needs inventories) builds the SHAP effect-agreement reference;
Stage B (needs no labels) carries it into a district that has no inventory.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from rev_common import ROOT

FIG = ROOT / "outputs" / "no_roads" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight"})

INK, MUTED, LINE = "#13120f", "#5f5d57", "#8c8a83"
DATA, DATA_E = "#e8f1fb", "#2a78d6"
COMP, COMP_E = "#e9f6f1", "#179c6d"
OUT,  OUT_E = "#fdf0e8", "#dd6a2f"

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.axis("off")
ax.set_xlim(0, 144)
ax.set_ylim(0, 84)


def node(x, y, w, h, n, title, sub, fc, ec, narrow=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=fc, ec=ec, lw=0.9, zorder=3))
    if narrow:
        ax.add_patch(Circle((x + 3.6, y + h - 3.2), 2.2, fc=ec, ec="none", zorder=4))
        ax.text(x + 3.6, y + h - 3.2, str(n), ha="center", va="center",
                fontsize=5.5, weight="bold", color="white", zorder=5)
        ax.text(x + w / 2, y + h - 7.4, title, ha="center", va="center", fontsize=6.8,
                weight="bold", color=INK, zorder=4)
        if sub:
            ax.text(x + w / 2, y + h - 11.4, sub, ha="center", va="center", fontsize=5.7,
                    color=MUTED, zorder=4)
        return
    cy = y + h - 4.6 if sub else y + h / 2
    ax.add_patch(Circle((x + 4.6, cy), 2.5, fc=ec, ec="none", zorder=4))
    ax.text(x + 4.6, cy, str(n), ha="center", va="center", fontsize=5.9, weight="bold",
            color="white", zorder=5)
    ax.text(x + 9.5, cy, title, ha="left", va="center", fontsize=7.3, weight="bold", color=INK, zorder=4)
    if sub:
        ax.text(x + 9.5, y + h - 9.8, sub, ha="left", va="center", fontsize=6.2, color=MUTED, zorder=4)


def vdown(x, y_top, y_bot):
    ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bot), arrowstyle="-|>", mutation_scale=8,
                                 color=LINE, lw=1.1, shrinkA=0, shrinkB=0, zorder=2))


# ---------------------------------------------------------------- stage panels
for x0, w0 in ((1, 63), (74, 69)):
    ax.add_patch(FancyBboxPatch((x0, 2), w0, 80, boxstyle="round,pad=0,rounding_size=2",
                                fc="#fbfaf8", ec="#dcdad3", lw=0.8, zorder=0))
    ax.add_patch(FancyBboxPatch((x0, 74), w0, 8, boxstyle="round,pad=0,rounding_size=2",
                                fc="#3d3b36", ec="none", zorder=1))

ax.text(32.5, 78.0, "STAGE A  ·  needs landslide inventories", ha="center", va="center",
        fontsize=7.5, weight="bold", color="white", zorder=2)
ax.text(108.5, 78.0, "STAGE B  ·  needs no landslide labels", ha="center", va="center",
        fontsize=7.5, weight="bold", color="white", zorder=2)
ax.text(32.5, 70.3, "done once, in regions that have an inventory", ha="center", va="center",
        fontsize=6.2, color=MUTED, style="italic", zorder=2)
ax.text(108.5, 70.3, "done for each new district to be mapped", ha="center", va="center",
        fontsize=6.2, color=MUTED, style="italic", zorder=2)

# ---------------------------------------------------------------- stage A
node(5, 51, 26, 14, 1, "Source region 1", "inventory + factors", DATA, DATA_E, narrow=True)
node(34, 51, 26, 14, 2, "Source region 2", "inventory + factors", DATA, DATA_E, narrow=True)
node(5, 35, 55, 12, 3, "Train a model per region", "fitted to its own inventory", COMP, COMP_E)
node(5, 21, 55, 10, 4, "Compare on the same cells", None, COMP, COMP_E)
node(5, 6, 55, 12, 5, "SHAP effect agreement", "the two models compared", OUT, OUT_E)

vdown(18, 50.6, 47.4)
vdown(47, 50.6, 47.4)
vdown(32.5, 34.6, 31.4)
vdown(32.5, 20.6, 18.4)

# ---------------------------------------------------------------- stage B
node(78, 51, 61, 14, 6, "Target district, no inventory", "factor rasters + background sample", DATA, DATA_E)
node(78, 35, 61, 12, 7, "Adversarial validation", "covariates only, labels in neither", COMP, COMP_E)
node(78, 20, 61, 12, 8, "Keep factors passing both", "unshifted and agreeing factors", COMP, COMP_E)
node(78, 6, 61, 10, 9, "Transferred susceptibility map", None, OUT, OUT_E)

vdown(108.5, 50.6, 47.4)
vdown(108.5, 34.6, 32.4)
vdown(108.5, 19.6, 18.4)

# ---------------------------------------------------------------- handover A -> B
ax.plot([60.3, 69], [12, 12], color=LINE, lw=1.1, zorder=2, solid_capstyle="round")
ax.plot([69, 69], [12, 26], color=LINE, lw=1.1, zorder=2, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((69, 26), (77.6, 26), arrowstyle="-|>", mutation_scale=8,
                             color=LINE, lw=1.1, shrinkA=0, shrinkB=0, zorder=2))
ax.text(69, 29.0, "agreement\nvalues", ha="center", va="bottom", fontsize=5.8,
        color=MUTED, linespacing=1.35, zorder=4)

fig.savefig(FIG / "Figure12_workflow_stages.png")
plt.close(fig)
print("Figure 12 saved")
