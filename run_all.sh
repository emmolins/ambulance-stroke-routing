#!/usr/bin/env bash
# =============================================================================
# run_all.sh — regenerate every result for the T-ASE paper, in order, with logs.
#
#   ./run_all.sh ca            # full California chain
#   ./run_all.sh ri            # Rhode Island replicates + stats (needs ORS-RI on $ORS_PORT)
#   START=5 ./run_all.sh ca    # resume from step 5
#   PAR=4 ./run_all.sh ca      # replicates in parallel (default 4)
#
# Stops at the first failing step. Logs: run_logs/<region>_<step>_<name>.log
# Keep the Mac awake:  caffeinate -i ./run_all.sh ca
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
REGION="${1:-ca}"; REGION_UC=$(echo "$REGION" | tr a-z A-Z)
START="${START:-1}"; PAR="${PAR:-4}"
mkdir -p run_logs simulation_results decision_tree_output
J="julia --project=."

step() {  # step <n> <name> <command...>
    local n=$1 name=$2; shift 2
    [[ $n -lt $START ]] && { echo "[skip] $n $name"; return; }
    local log="run_logs/${REGION}_$(printf %02d "$n")_${name}.log"
    echo "[$(date +%H:%M)] step $n: $name  -> $log"
    if ! "$@" > "$log" 2>&1; then
        echo "[FAIL] step $n ($name). Last lines:"; tail -20 "$log"
        echo "Resume with:  START=$n ./run_all.sh $REGION"; exit 1
    fi
    echo "[ok]   step $n: $name"
}

replicates() {  # replicates <driver> <extra env...>  -> 10 reps, $PAR at a time
    local driver=$1; shift
    seq 1 10 | xargs -P "$PAR" -I{} sh -c \
        "env $* REPLICATE_INDEX={} $J '$driver' > 'run_logs/${REGION}_rep_{}.log' 2>&1"
}

if [[ $REGION == ca ]]; then
    step 1  tree_build        $J decision_tree/decision_tree_build.jl
    step 2  replicates        replicates src/CA_simulations.jl
    step 3  aggregate         $J scripts/aggregate_replicates.jl
    step 4  realized          $J scripts/recompute_realized_rewards.jl
    step 5  stats_marginal    $J src/CA_simulations_stats.jl
    step 6  stats_realized    env INPUT_PREFIX=CA_simulation_results_realized $J src/CA_simulations_stats.jl
    step 7  perturb_noise10   env TRAVEL_TIME_NOISE_SD=0.10 $J scripts/recompute_perturbed_rewards.jl
    step 8  perturb_noise20   env TRAVEL_TIME_NOISE_SD=0.20 $J scripts/recompute_perturbed_rewards.jl
    step 9  perturb_bias      bash -c 'for b in 0.70 0.80 0.90 1.10 1.20 1.30; do TRAVEL_TIME_BIAS_MULT=$b julia --project=. scripts/recompute_perturbed_rewards.jl || exit 1; done'
    step 10 aggregate_sens    $J scripts/aggregate_replicates.jl
    step 11 sensitivity       $J scripts/sensitivity_summary.jl
    step 12 dido_sens         bash -c 'for d in 60 90 175; do DIDO_MIN=$d REALIZED_TAG=_dido$d julia --project=. scripts/recompute_realized_rewards.jl || exit 1; done'
    step 13 evt_onsite_reps   bash -c 'for i in 1 2 3; do EVT_DEFINITION=onsite OUTPUT_TAG=_evtonsite REPLICATE_INDEX=$i julia --project=. src/CA_simulations.jl || exit 1; done'
    step 14 evt_onsite_pool   $J -e 'using CSV, DataFrames; fs = sort(filter(f -> occursin(r"^CA_simulation_results_rep\d+_evtonsite\.csv$", f), readdir("simulation_results"))); dfs = [begin d = CSV.read(joinpath("simulation_results", f), DataFrame); d.replicate .= parse(Int, match(r"_rep(\d+)_", f).captures[1]); d end for f in fs]; CSV.write("simulation_results/CA_simulation_results_evtonsite.csv", vcat(dfs...)); println(length(fs), " files pooled")'
    step 15 evt_onsite_stats  env INPUT_PREFIX=CA_simulation_results_evtonsite $J src/CA_simulations_stats.jl
    step 16 tree_sweep        $J decision_tree/decision_tree_sweep.jl
    step 17 combine_paper     $J scripts/combine_for_paper.jl
    step 18 grid_generator    env N_PER_CELL=5 $J scripts/grid_generator.jl
    step 19 grid_aggregate    $J scripts/grid_aggregate.jl
    step 20 grid_tract_join   data_prep/.venv/bin/python scripts/grid_to_tract_join.py
    step 21 equity            $J scripts/equity_analysis.jl
    step 22 grid_maps         $J viz/CA_grid_maker.jl
    step 23 latency           $J scripts/computational_analysis.jl
elif [[ $REGION == ri ]]; then
    export ORS_PORT="${ORS_PORT:-8081}"
    step 1  replicates        replicates src/RI_simulations.jl
    step 2  aggregate         env REGION=RI $J scripts/aggregate_replicates.jl
    step 3  stats_marginal    $J src/RI_simulations_stats.jl
else
    echo "usage: $0 ca|ri"; exit 2
fi
echo "[$(date +%H:%M)] all steps done for $REGION"
