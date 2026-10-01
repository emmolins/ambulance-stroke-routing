# SETUP — Julia environment, VS Code, and end-to-end run guide

This is the "get the code running" guide. ORS (the external routing server the simulations call) has its own setup in `ORS_SETUP.md`. Data-prep notebooks have their own README in `data_prep/`.

## 1. Install Julia

Recommended (handles versions cleanly):

```sh
curl -fsSL https://install.julialang.org | sh
```

Open a new terminal afterward and verify:

```sh
julia --version
```

## 2. Install VS Code and the Julia extension

- VS Code: https://code.visualstudio.com
- Install the `code` shell command: in VS Code, `Cmd+Shift+P` → "Shell Command: Install 'code' command in PATH".
- VS Code will auto-prompt to install the recommended Julia extension when you open this folder (see `.vscode/extensions.json`). Accept.

## 3. Set up the Julia environment

```sh
cd ~/StrokeRouting/"Stroke Routing"
julia --project=. setup.jl
```

First run takes several minutes (downloading + precompiling ~22 packages). After it finishes you'll have pinned `Project.toml` + `Manifest.toml` files. From then on, always start Julia in this repo with:

```sh
julia --project=.
```

## 4. Open in VS Code

```sh
cd ~/StrokeRouting/"Stroke Routing"
code .
```

VS Code will detect `Project.toml` and use it as the active Julia environment.

## 5. Repository layout

```
Stroke Routing/
├── src/                            Core MDP models, simulation drivers, stats
│   ├── CA_STPMDP_ORS.jl                California MDP + reward function
│   ├── RI_STPMDP_ORS.jl                Rhode Island MDP
│   ├── CA_simulations.jl               Driver: 1 replicate × 1000 patients
│   ├── RI_simulations.jl               (parallel for RI)
│   ├── CA_simulations_stats.jl         Paired t-tests, subgroups, between-rep CIs
│   └── RI_simulations_stats.jl
├── scripts/                        Post-hoc analyses and pipeline glue
│   ├── aggregate_replicates.jl         Pool per-replicate CSVs into one file
│   ├── recompute_realized_rewards.jl   Switch reward semantics: UNKNOWN → KNOWN
│   ├── recompute_perturbed_rewards.jl  Post-hoc travel-time sensitivity
│   ├── sensitivity_summary.jl          Build the noise/bias robustness table
│   ├── combine_for_paper.jl            Manuscript-ready combined tables
│   ├── computational_analysis.jl       best_action latency vs depth
│   ├── grid_generator.jl               Per-cell grid simulations (equity)
│   ├── grid_aggregate.jl               Pool grid cells
│   ├── grid_to_tract_join.py           Spatial join to Census tracts
│   └── equity_analysis.jl              Stratified by income + urbanicity
├── decision_tree/                  Interpretable-policy distillation
│   ├── decision_tree_utils.jl
│   ├── decision_tree_build.jl          Generates training dataset
│   ├── decision_tree_eval.jl           Evaluates a trained tree
│   ├── decision_tree_sweep.jl          Depth-vs-accuracy sweep (paper figure)
│   └── decision_tree_styler.jl         TikZ styling helper
├── viz/                            Visualization
│   └── CA_grid_maker.jl                Bay Area outcome heatmaps
├── data_prep/                      Patient sampling + equity overlay
│   ├── filter_points_CA.ipynb          Generate canonical CA patient pool
│   ├── filter_points_RI.ipynb          (parallel for RI)
│   ├── build_bay_area_grid.py          200×200 cell mask for the 9-county region
│   ├── build_acs_overlay.py            ACS demographics + USDA RUCA codes
│   ├── README.md
│   └── requirements.txt
├── hospitals/                      Hospital metadata
│   ├── CA_hospitals.csv                v4: 61 facilities, sanitized identifiers
│   └── RI_hospitals.csv
├── sampled_points/                 Canonical patient pools + grid mask
├── simulation_results/             Outputs (gitignored)
├── decision_tree_output/           Outputs (gitignored)
├── setup.jl                        One-shot env bootstrap
├── Project.toml                    Pinned dependencies
├── Manifest.toml                   Locked versions
├── LICENSE                         MIT
├── SETUP.md                        (this file)
└── ORS_SETUP.md                    OpenRouteService Docker setup
```

All commands below are run **from the repo root**:

```sh
cd ~/StrokeRouting/"Stroke Routing"
```

## 6. Smoke test (no ORS required)

```sh
julia --project=. -e 'include("src/CA_STPMDP_ORS.jl"); mdp = StrokeMDP(); println("loaded $(length(mdp.locations)) hospitals")'
```

Expected output: `loaded 61 hospitals`. Anything that hits ORS requires the local ORS server — see `ORS_SETUP.md`.

## 7. Reproducing the full California analysis

The complete pipeline from a clean clone to manuscript-ready tables and figures. Items in `[brackets]` are wall-clock estimates on a commodity laptop.

### 7.1. Primary results — 10 replicates × 1000 patients [~4-5 h]

```sh
for i in 1 2 3 4 5 6 7 8 9 10; do
    REPLICATE_INDEX=$i nohup julia --project=. src/CA_simulations.jl \
        > "simulation_results/CA_run_rep${i}.log" 2>&1 &
done
wait
```

### 7.2. Pool + run primary stats [~5 min]

```sh
julia --project=. scripts/aggregate_replicates.jl
julia --project=. src/CA_simulations_stats.jl
```

Produces `simulation_results/CA_paired_comparisons.csv` (within-replicate CIs) and `CA_paired_comparisons_multirep.csv` (between-replicate CIs — the IEEE-canonical version).

### 7.3. Realized rewards [~5 min]

The planner uses marginalized expectation at decision time. The realized reward conditions on each patient's actual stroke type — the standard clinical-paper metric. The planner is unchanged.

```sh
julia --project=. scripts/recompute_realized_rewards.jl
INPUT_PREFIX=CA_simulation_results_realized julia --project=. src/CA_simulations_stats.jl
julia --project=. scripts/combine_for_paper.jl
```

### 7.4. Travel-time sensitivity [~10 min]

Post-hoc perturbation of recorded travel times. Tests robustness to (a) variance and (b) systematic ORS calibration bias.

```sh
# Variance: multiplicative log-normal noise
TRAVEL_TIME_NOISE_SD=0.10 julia --project=. scripts/recompute_perturbed_rewards.jl
TRAVEL_TIME_NOISE_SD=0.20 julia --project=. scripts/recompute_perturbed_rewards.jl

# Systematic bias: ±10/20/30% ORS-vs-real-EMS calibration error
for b in 0.70 0.80 0.90 1.10 1.20 1.30; do
    TRAVEL_TIME_BIAS_MULT=$b julia --project=. scripts/recompute_perturbed_rewards.jl
done

# Pool + summary table
julia --project=. scripts/aggregate_replicates.jl
julia --project=. scripts/sensitivity_summary.jl
```

Output: `simulation_results/CA_sensitivity_summary.csv` (one row per setting × comparison).

### 7.5. Computational analysis [~5-10 min]

Times `best_action()` at depths {1, 2, 3, 4} over 50 representative patient states. Used for the deployability argument.

```sh
julia --project=. scripts/computational_analysis.jl
```

Output: `simulation_results/CA_computational_analysis.csv` + `CA_computational_summary.csv`.

### 7.6. Decision-tree distillation [~1-2 h]

```sh
julia --project=. decision_tree/decision_tree_build.jl   # training data + base tree
julia --project=. decision_tree/decision_tree_sweep.jl   # depth-vs-accuracy sweep
```

Outputs in `decision_tree_output/`: training data, sweep summary, tradeoff figure.

### 7.7. Equity analysis (200×200 Bay Area grid) [~35-50 h grid + ~5 min downstream]

```sh
# One-time prep (~10 min, do once)
cd data_prep
.venv/bin/pip install -r requirements.txt
.venv/bin/python build_bay_area_grid.py
export CENSUS_API_KEY=YOUR-FREE-KEY   # https://api.census.gov/data/key_signup.html
.venv/bin/python build_acs_overlay.py
cd ..

# The long run (~35-50 h, resumable — kill & restart any time)
nohup julia --project=. scripts/grid_generator.jl \
    > simulation_results/grid_generator.log 2>&1 &

# After grid completes (~5 min total)
julia --project=. scripts/grid_aggregate.jl
data_prep/.venv/bin/python scripts/grid_to_tract_join.py
julia --project=. scripts/equity_analysis.jl
```

### 7.8. Heatmap figures

```sh
julia --project=. viz/CA_grid_maker.jl
```

## 8. Reproducing the Rhode Island analysis

Same shape as CA, with `RI_` prefixes:

```sh
for i in 1 2 3 4 5 6 7 8 9 10; do
    REPLICATE_INDEX=$i nohup julia --project=. src/RI_simulations.jl \
        > "simulation_results/RI_run_rep${i}.log" 2>&1 &
done
wait
julia --project=. src/RI_simulations_stats.jl
```

The post-hoc sensitivity, decision-tree, and grid pipelines are CA-specific in the current code.

## 9. Notes for reviewers

- All randomness is seeded (`BASE_SEED = 1234` in `src/CA_simulations.jl`; per-replicate seeds derive from `REPLICATE_INDEX`). A clean rerun reproduces results bit-exactly given the pinned `Manifest.toml`.
- Patient sampling is without replacement from a canonical 10 000-point pool generated in `data_prep/filter_points_CA.ipynb`.
- The MDP planner sees UNKNOWN stroke type at decision time (correct decision-theoretic setup). Realized outcomes condition on each patient's true sampled type via the post-hoc reward recomputation.
- Travel-time sensitivity is applied *post-hoc* on the simulation's recorded travel times (decision-first, realization-after) — the planner never sees the perturbation.
- Generated outputs (`simulation_results/`, `decision_tree_output/`, per-cell grid CSVs) are gitignored. The repo ships the code to reproduce them, not the outputs themselves.
