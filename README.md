# Ambulance Routing for Stroke Triage

A Markov Decision Process (MDP) approach to dispatching ambulances carrying suspected stroke patients to the right hospital — Comprehensive Stroke Center (CSC), Primary Stroke Center (PSC), or Clinic — given the patient's location, time since symptom onset, and the regional hospital network.

Two regions are analyzed:

- **California**, with a focus on the nine-county San Francisco Bay Area (61 stroke-receiving hospitals).
- **Rhode Island**, as a smaller-region generalizability test.

For each region, the MDP-optimal policy is compared against the current dispatch convention (nearest-hospital routing) and two clinically motivated heuristic policies. A decision-tree distillation provides an interpretable approximation of the MDP for potential clinical deployment.

This codebase accompanies a manuscript currently under revision. The full reproduction pipeline — from patient sampling through manuscript-ready tables — is documented in [`SETUP.md`](SETUP.md).

## Headline analyses

- **Primary results**: paired t-tests of expected good-outcome probability across all four policies (MDP-optimal vs nearest, vs heuristic 1, vs heuristic 2), with between-replicate 95% CIs computed across 10 independent replicates of 1,000 patients each.
- **Travel-time sensitivity**: post-hoc perturbation of recorded travel times under multiplicative log-normal noise (σ ∈ {0.10, 0.20}) and systematic ORS-vs-real-EMS bias (×0.70 to ×1.30). Tests robustness of the policy ranking without rebuilding the planner.
- **Equity analysis**: 200×200 outcome grid over the 9-county Bay Area, spatially joined to Census tract demographics (median household income from ACS 2022 5-year) and USDA RUCA urban/rural codes. Stratified policy contrasts across income quartiles and urbanicity.
- **Decision-tree distillation**: interpretable rules trained against MDP-optimal labels, with a depth-vs-accuracy sweep to characterize the cost of interpretability.
- **Computational analysis**: empirical timing of `best_action()` across forward-search depths, supporting the real-time deployability argument.

## Repository at a glance

```
src/            MDP model, simulation driver, and primary stats   (Julia)
scripts/        Post-hoc analyses and pipeline glue                (Julia + Python)
decision_tree/  Interpretable-policy distillation                  (Julia)
data_prep/      Patient sampling + equity overlay                  (Python notebooks/scripts)
viz/            Heatmap figures                                    (Julia)
hospitals/      Hospital metadata (61 CA + RI)
sampled_points/ Canonical patient pools + Bay Area grid mask
```

Full directory tree, file-by-file descriptions, and exact run commands: [`SETUP.md`](SETUP.md).

## Quick start

```sh
# 1. Julia env (one time, ~5 min)
curl -fsSL https://install.julialang.org | sh
cd ~/StrokeRouting/"Stroke Routing"
julia --project=. setup.jl

# 2. Bring up the OpenRouteService Docker server (one time, ~30 min for the graph build)
#    See ORS_SETUP.md.

# 3. MDP smoke test (no ORS required)
julia --project=. -e 'include("src/CA_STPMDP_ORS.jl"); mdp = StrokeMDP(); println(length(mdp.locations))'
#    → 61
```

Anything beyond the smoke test requires the local ORS server. See [`ORS_SETUP.md`](ORS_SETUP.md) for the Docker setup, and [`SETUP.md`](SETUP.md) sections 7–8 for the full analysis pipeline.

## External data sources

| Source | Used for | Where to find it |
|---|---|---|
| OpenRouteService v8 (Docker) | Travel-time matrix between any two points | Local instance — see `ORS_SETUP.md` |
| California OSM extract (Geofabrik) | ORS road graph for CA | https://download.geofabrik.de/ |
| TIGER 2020 census tracts | Tract polygons (equity overlay) | Auto-downloaded by `data_prep/build_acs_overlay.py` |
| ACS 2022 5-year demographics | Median household income, poverty rate | Census API, requires free key |
| USDA ERS RUCA 2010 | Urban/rural classification | Auto-downloaded by `data_prep/build_acs_overlay.py` |
| Bay Area ZIP shapefile | 9-county boundary mask | Berkeley GeoData library |
| HDX/Meta population density | Patient-location sampling weights | `data_prep/data/` (see `data_prep/README.md`) |

## Reproducibility

- All RNG is seeded. The base seed is set in `src/CA_simulations.jl` (`BASE_SEED = 1234`); per-replicate seeds derive deterministically from `REPLICATE_INDEX`. A clean rerun against the pinned `Manifest.toml` is bit-exact.
- Patient locations are drawn without replacement from a canonical 10,000-point pool, generated in `data_prep/filter_points_CA.ipynb`. The canonical CSV is committed; the underlying raster/shapefile inputs are listed in `.gitignore` with download instructions in `data_prep/README.md`.
- Generated outputs (`simulation_results/`, `decision_tree_output/`, per-cell grid CSVs) are gitignored. The repo ships the code to reproduce them, not the outputs themselves.

## References

The reward function and stroke-outcome curves are taken from:

Holodinsky JK, Williamson TS, Demchuk AM, Zhao H, Zhu L, Francis MJ, Goyal M, Hill MD, Kamal N. *Modeling stroke patient transport for all patients with suspected large-vessel occlusion.* JAMA Neurology. 2018;75(12):1477–1486. https://doi.org/10.1001/jamaneurol.2018.2424

## License

MIT — see [`LICENSE`](LICENSE).
