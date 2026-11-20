"""R4: robustness of the SHAP effect-agreement metric.
Pearson vs Spearman vs Lin's concordance; two TreeSHAP algorithms; three evaluation samples;
bootstrap confidence intervals over absence draws. Answers Reviewer 2 point 6.
-> outputs/revision/shap_metric_robustness.csv
"""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import shap
from rev_common import *

S = load("no_roads")
FE = feature_sets(S)["S1_all"]
OH = onehot(S)
GROUP = {**{c: c for c in CONT}, **{c: "landcover" for c in OH if c.startswith("lc_")},
         **{c: "lithology" for c in OH if c.startswith("lith_")}}


def ccc(a, b):
    """Lin's concordance correlation coefficient: agreement in value, not just in rank."""
    a, b = np.asarray(a), np.asarray(b)
    return float(2 * np.cov(a, b)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def grouped(sv, cols):
    return pd.DataFrame(sv, columns=cols).T.groupby(pd.Series(GROUP)[cols].values).sum().T


rows = []
for seed in SEEDS:
    fits, bg = {}, {}
    for reg in ("wayanad", "kodagu"):
        d = S[(S.district == reg) & (S.seed == seed)]
        fits[reg] = models(seed)["XGB"].fit(d[FE], d.label)
        bg[reg] = d
    pooled = S[S.seed == seed].reset_index(drop=True)
    evalsets = {"pooled": pooled, "wayanad_samples": bg["wayanad"].reset_index(drop=True),
                "kodagu_samples": bg["kodagu"].reset_index(drop=True)}
    bgref = pooled.sample(n=200, random_state=seed)[FE]
    for algo in ("tree_path_dependent", "interventional"):
        ex = {r: (shap.TreeExplainer(fits[r], feature_perturbation="tree_path_dependent") if algo == "tree_path_dependent"
                  else shap.TreeExplainer(fits[r], data=bgref, feature_perturbation="interventional"))
              for r in fits}
        for ename, e in evalsets.items():
            sv = {r: grouped(ex[r].shap_values(e[FE], check_additivity=False), FE) for r in fits}
            for f in sv["wayanad"].columns:
                a, b = sv["wayanad"][f].values, sv["kodagu"][f].values
                if a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                rows.append(dict(seed=seed, algo=algo, eval_sample=ename, factor=f,
                                 pearson=float(pearsonr(a, b)[0]), spearman=float(spearmanr(a, b)[0]),
                                 ccc=ccc(a, b)))
    print("shap robustness seed", seed, "done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / "shap_metric_runs.csv", index=False)


def boot_ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


out = []
for (algo, ename, f), g in R.groupby(["algo", "eval_sample", "factor"]):
    lo, hi = boot_ci(g.pearson.values)
    out.append(dict(algo=algo, eval_sample=ename, factor=f, pearson_mean=g.pearson.mean(),
                    pearson_sd=g.pearson.std(), pearson_lo95=lo, pearson_hi95=hi,
                    spearman_mean=g.spearman.mean(), ccc_mean=g.ccc.mean()))
O = pd.DataFrame(out).round(3)
O.to_csv(OUT / "shap_metric_robustness.csv", index=False)
main = O[(O.algo == "tree_path_dependent") & (O.eval_sample == "pooled")].sort_values("pearson_mean", ascending=False)
print(main.to_string())
print("\nrank correlation between metrics (pooled, tree_path_dependent):")
print("pearson vs spearman:", round(float(main[["pearson_mean", "spearman_mean"]].corr(method="spearman").iloc[0, 1]), 3),
      "| pearson vs ccc:", round(float(main[["pearson_mean", "ccc_mean"]].corr(method="spearman").iloc[0, 1]), 3))
