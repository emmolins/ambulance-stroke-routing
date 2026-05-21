# SETUP — Julia environment and VS Code

This is the "get the code running" guide. ORS (the external routing server the simulations call) has its own setup in `ORS_SETUP.md`.

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

The original repo had no `Project.toml`/`Manifest.toml`, which is a reproducibility gap for publication. The `setup.jl` script in this folder fixes that by activating a project-local environment, adding every package the source files import, and producing pinned `Project.toml` + `Manifest.toml` files.

```sh
cd ~/StrokeRouting/"Stroke Routing"
julia --project=. setup.jl
```

First run takes several minutes (downloading + precompiling ~22 packages including Plots, GMT, POMDPs, DecisionTree, TikzGraphs). After it finishes:

```sh
git add Project.toml Manifest.toml setup.jl SETUP.md ORS_SETUP.md .gitignore .vscode/
git commit -m "Add Julia env, VS Code config, and ORS setup docs for publication revisions"
```

From now on, always start Julia in this repo with:

```sh
julia --project=.
```

## 4. Open in VS Code

```sh
cd ~/StrokeRouting/"Stroke Routing"
code .
```

When VS Code opens:
- It will prompt you to install the Julia extension if you haven't already — accept.
- It will detect `Project.toml` and use it as the active Julia environment.
- Open any `.jl` file, then `Cmd+Shift+P` → "Julia: Start REPL" to bring up a REPL using the project's environment.

## 5. Smoke test (no ORS required)

To sanity-check that all source files at least parse and load:

```julia
# inside the Julia REPL, started with --project=.
include("decision_tree_utils.jl")
```

For scripts that actually query ORS (`CA_STPMDP_ORS.jl`, `RI_STPMDP_ORS.jl`, `CA_simulations.jl`, `RI_simulations.jl`, `decision_tree_build.jl`), start the ORS server first — see `ORS_SETUP.md`.

## 6. (Optional) publication-readiness cleanup

While in revision mode:

- `gmt.history` is GMT.jl's plotting cache; now in `.gitignore`. Consider `git rm --cached gmt.history` to drop it from the index.
- `simulation_results/` and `decision_tree_output/` contain checked-in PDFs/CSVs from prior runs. For a reproducibility-focused publication, consider moving them to a `paper-figures/` folder or regenerating them from the pinned `Manifest.toml`.
- Add a `LICENSE` file before publication.
- Consider a top-level `run_all.jl` that reproduces every figure in the paper from a clean clone — reviewers love these.
