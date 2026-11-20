# Landslide susceptibility transferability in the Western Ghats (Wayanad ↔ Kodagu)

Open-data Python workflow testing whether a machine-learning landslide susceptibility model, and its
explanations, survive a move into a neighbouring district that supplied no training data.

Wayanad (Kerala) and Kodagu (Karnataka) are adjoining Western Ghats districts, both severely affected by
the August 2018 monsoon, and each has an openly available landslide inventory. Four classifiers are
evaluated under three designs of increasing strictness — random cross-validation, 5 km spatial block
cross-validation, and cross-district transfer in both directions — so that the gap between what validation
promises and what transfer delivers can be measured directly. Two diagnostics are then used to explain
that gap: adversarial validation, which quantifies covariate shift and needs no landslide labels, and SHAP
**effect agreement**, which compares what two locally trained models attribute to the same factor values
rather than comparing importance rankings.

This repository reproduces every table and figure in the associated manuscript, from the raw open inputs
to the finished document.

## Repository layout

```
00_extract_osm_roads.py … 08_manuscript_tables.py   main pipeline, in run order
rev_common.py, rev_01_*.py … rev_13_*.py            analyses added during peer review
config.py                                           paths, CRS, factor lists, lithology/land-cover maps
requirements.txt
data/processed/                                     derived sample tables (see below)
outputs/no_roads/, outputs/with_roads/              results, figures and maps per variant
outputs/revision/                                   results of the rev_* scripts
manuscript/tables.json, numbers.json                every number as typeset in the article
```

`data/raw/` is not included; `01_clip_inputs.py` downloads or reads the original open datasets and clips
them to each district.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. The main dependencies are rasterio, GeoPandas, shapely, pysheds, scikit-learn, XGBoost,
LightGBM, SHAP, statsmodels, pandas, NumPy, SciPy and matplotlib; `00_extract_osm_roads.py` additionally
needs pyosmium. With NumPy ≥ 2, pysheds needs the shim `np.in1d = np.isin` before import, which
`02_terrain_hydrology.py` applies.

## Reproducing the study

```bash
python 01_clip_inputs.py          # clip and align all factors to a 30 m UTM 43N grid
python 02_terrain_hydrology.py    # slope, aspect, curvature, TRI, TWI, SPI, distance to streams
python 03_sample.py               # ten presence-absence draws per district + background samples
python 04_models.py               # LR, RF, XGBoost, LightGBM under the three validation designs
python 05_maps.py                 # ensemble susceptibility maps, local and transferred
python 06_figures.py              # Figures 1-8
python 07_supplementary_stats.py
python 08_manuscript_tables.py
```

For the roads sensitivity analysis, run `00_extract_osm_roads.py` once and then repeat the pipeline with
`USE_ROADS=1`, which adds distance to roads to the factor list and writes to `outputs/with_roads/`.
`FORCE=1` re-derives rasters that already exist.

### Analyses added during peer review

These reuse the same samples, classifiers and protocol, so run `03_sample.py` and `04_models.py` first.
`rev_common.py` holds the shared helpers (factor sets, 5 km blocks, spatial-CV prediction, AUC on
slope-restricted subsets, Benjamini–Hochberg correction).

| Script | What it does | Output in the article |
|---|---|---|
| `rev_01_factorsets.py` | transfer under factor subsets T1, T2, T3; steep-terrain evaluation | Tables 9, 10 |
| `rev_02_adversarial.py` | decomposition of the multivariate adversarial AUC; greedy forward selection; TreeSHAP on the adversarial classifier | Table 8 |
| `rev_03_inventory.py` | inventory geometry (density, nearest neighbour, Clark–Evans, KS tests) and the thinning experiment | Tables 11, 12 |
| `rev_04_shap_robust.py` | robustness of effect agreement: Pearson, Spearman, Lin's CCC; two TreeSHAP algorithms; three evaluation samples; bootstrap CIs | Table S3 |
| `rev_05_tuning.py` | randomised hyperparameter search scored by spatial block CV, applied unchanged to transfer | Table S4 |
| `rev_06_rainfall.py` | location-code probes: rainfall replaced by a spatial trend; rainfall-only and trend-only models; slope and TRI coarsened to the 0.05° grid | Table 13 |
| `rev_07_roads.py` | distance-to-roads diagnostics: effect agreement and distance summaries | Table S5 |
| `rev_08_fdr_steep.py` | Benjamini–Hochberg correction of every paired Wilcoxon test; steep-terrain summary | Table S2 |
| `rev_09_tables.py` | builds `manuscript/tables.json` and `numbers.json` | Tables 8–13, S2–S5 |
| `rev_10_figures.py`, `rev_11_figures.py`, `rev_12_fig2.py`, `rev_13_fig12.py` | figures | Figures 2, 9, 10, 11, 12 |

Run them in numerical order; `rev_09` and the figure scripts depend on the CSVs written by `rev_01`–`rev_08`.

## Data

| Dataset | Source |
|---|---|
| Kerala 2018 landslide inventory | https://doi.org/10.17026/dans-x6c-y7x2 |
| Kodagu 2018 landslide inventory | https://github.com/Arpithaachaiah6/Landslide-inventory-for-the-2018-storm-event-of-Kodagu-in-the-Western-Ghats |
| Copernicus GLO-30 DEM | https://doi.org/10.5270/ESA-c5d3d65 |
| CHIRPS v2.0 rainfall | https://data.chc.ucsb.edu/products/CHIRPS-2.0/ |
| SoilGrids 250 m | https://soilgrids.org |
| ESA WorldCover 2021 | https://doi.org/10.5281/zenodo.7254221 |
| GSI district geology / National Geoscience Data Repository | http://geodataindia.gov.in/ |
| geoBoundaries ADM2 | https://www.geoboundaries.org |
| OpenStreetMap roads | https://download.geofabrik.de |

**What is included here.** `outputs/` holds the derived results on which every reported statistic is
computed, and `data/processed/` holds the sample tables the models are fitted to: `samples_no_roads.csv`
and `samples_with_roads.csv` (the ten presence-absence draws per district) and `background_no_roads.csv.gz`
(20,000 background cells per district, used for the shift diagnostics and regional standardisation). The
analysis can therefore be reproduced without re-downloading the raw rasters.

**What is not.** The processed 30 m factor rasters are not redistributed, because they are derivatives of
the third-party datasets above. `01_clip_inputs.py` and `02_terrain_hydrology.py` rebuild them from the
original sources with the parameters fixed in `config.py`.

## Notes

- Hyperparameters are fixed a priori for the main results; `rev_05_tuning.py` tests what tuning changes.
- Maps of a training district are fitted to its own inventory and are shown for comparison only; the
  transferred maps are the independent tests.
- AUC computed over a whole district rewards the separation of steep ground from gentle plateau;
  `rev_01_factorsets.py` repeats every evaluation on subsets restricted to slopes above 10°, 15°, 20° and 25°.
- Results are reported as mean ± standard deviation over the ten absence draws, and paired tests are
  descriptive summaries of consistency across draws rather than confirmatory inference.

## Citation

Kellengere Shankarnarayan, V. What transfers and what does not: covariate shift, SHAP explanation stability
and cross-district transferability of machine-learning landslide susceptibility models in the Western Ghats,
India. Manuscript under review, *Frontiers in Earth Science*.

Archived version of this repository: [Zenodo DOI to be inserted].

## Licence

Code in this repository is released under the MIT licence (see `LICENSE`). The input datasets keep their own
licences and are not redistributed here.
