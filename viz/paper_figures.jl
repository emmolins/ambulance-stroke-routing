# =============================================================================
# viz/paper_figures.jl — publication figures for the T-ASE manuscript (Plots.jl, GR).
#
#   julia --project=. viz/paper_figures.jl            # all figures
#   FIG=onset julia --project=. viz/paper_figures.jl  # one figure: onset|space|dido|forest|load|map
#
# Reads simulation_results/*.csv and decision_tree_output/*.csv, writes vector PDFs
# to paper/figures/. Style: IEEE single column 3.5 in (252 pt) or double 7.16 in,
# 8 pt serif type, one colour per policy throughout, colourblind-safe.
# =============================================================================
using CSV, DataFrames, Statistics, StatsBase, JSON, Plots
gr()

const OUT = "paper/figures"; mkpath(OUT)
const SR  = "simulation_results"
const PT  = 72 / 25.4          # points per mm (unused helper)
const W1  = 252                # single column, pt
const W2  = 516                # double column, pt
const FIG = lowercase(get(ENV, "FIG", "all"))

# ---- policy palette (fixed assignment, used in every figure)
const COL = Dict(:mdp => RGB(42/255, 120/255, 214/255),   # blue
                 :h1  => RGB(235/255, 104/255, 52/255),   # orange, nearest EVT-capable
                 :h2  => RGB(27/255, 175/255, 122/255),   # teal, nearest stroke center
                 :nh  => RGB(110/255, 110/255, 110/255))  # grey, nearest hospital
const LAB = Dict(:mdp => "MDP policy", :h1 => "Nearest EVT-capable center",
                 :h2 => "Nearest stroke center", :nh => "Nearest hospital")
const RW  = Dict(:mdp => :optimal_action_reward, :nh => :nearest_hospital_reward,
                 :h1 => :heuristic_1_reward, :h2 => :heuristic_2_reward)
const ACT = Dict(:mdp => :optimal_action, :nh => :nearest_hospital_action,
                 :h1 => :heuristic_1_action, :h2 => :heuristic_2_action)

default(fontfamily = "Computer Modern", guidefontsize = 8, tickfontsize = 7, legendfontsize = 7,
        titlefontsize = 8, framestyle = :box, grid = true, gridalpha = 0.25, gridlinewidth = 0.4,
        foreground_color_axis = :gray40, foreground_color_border = :gray40, linewidth = 1.5,
        markerstrokewidth = 0, legend_background_color = :transparent, legend_foreground_color = :transparent,
        dpi = 300, left_margin = 2Plots.mm, bottom_margin = 2Plots.mm, top_margin = 1Plots.mm, right_margin = 2Plots.mm)

# in = 72 pt; Plots sizes are in px at dpi 100 for the on-screen size; we size in pt directly
sz(wpt, hpt) = (round(Int, wpt * 100 / 72), round(Int, hpt * 100 / 72))

d = CSV.read(joinpath(SR, "CA_simulation_results.csv"), DataFrame)
# first split of the fitted tree (minutes since onset), read from decision_tree_simple.txt
const TREE_SPLIT = let f = "decision_tree_output/decision_tree_simple.txt"
    m = isfile(f) ? match(r"Time since onset\s*[≤<=]+\s*([0-9.]+)", read(f, String)) : nothing
    m === nothing ? 165.0 : round(parse(Float64, m.captures[1]))
end
println("CA results: ", nrow(d), " patients, ", length(unique(d.replicate)), " replicates")

# between-replicate mean and t_9 half-width of a paired difference
function rep_ci(diff::AbstractVector, rep::AbstractVector)
    per = [mean(diff[rep .== r]) for r in sort(unique(rep))]
    m = mean(per); se = std(per) / sqrt(length(per))
    t = Dict(9 => 2.262, 2 => 4.303, 4 => 2.776, 5 => 2.571)[max(length(per) - 1, 2)]
    return m, t * se
end

# ---------------------------------------------------------------------------
# Fig. onset: paired gain over nearest-hospital routing by onset-to-pickup time
# ---------------------------------------------------------------------------
function fig_onset()
    edges = 30:30:270; mids = collect(edges[1:end-1]) .+ 15
    p = plot(size = sz(W1, 170), xlabel = "Onset-to-pickup time (min)",
             ylabel = "Gain over nearest-hospital routing\nP(mRS 0–1)", legend = :topright, xlims = (30, 270))
    hline!(p, [0], color = :gray40, linewidth = 0.6, label = "")
    for (k, ls, lw) in ((:h2, :solid, 1.2), (:h1, :dash, 1.2), (:mdp, :solid, 2.0))
        diff = d[!, RW[k]] .- d[!, RW[:nh]]
        m = Float64[]; h = Float64[]
        for (a, b) in zip(edges[1:end-1], edges[2:end])
            sel = (d.t_onset .>= a) .& (d.t_onset .< b)
            mm, hh = rep_ci(diff[sel], d.replicate[sel]); push!(m, mm); push!(h, hh)
        end
        plot!(p, mids, m, ribbon = h, fillalpha = 0.15, color = COL[k], linestyle = ls, linewidth = lw,
              marker = :circle, markersize = 2.5, label = LAB[k])
    end
    vline!(p, [TREE_SPLIT], color = :gray40, linestyle = :dot, linewidth = 0.6, label = "")
    annotate!(p, TREE_SPLIT + 3, 0.028, text("tree split\n$(Int(TREE_SPLIT)) min", 6, :gray30, :left))
    savefig(p, joinpath(OUT, "fig_gain_by_onset.pdf")); println("✓ fig_gain_by_onset.pdf")
end

# ---------------------------------------------------------------------------
# Fig. space: the MDP destination tier in the two dispatcher variables
# ---------------------------------------------------------------------------
function fig_space()
    tr = CSV.read("decision_tree_output/training_data_detailed.csv", DataFrame)
    tier = Dict("Route_CSC" => ("EVT-capable center", COL[:mdp]),
                "Route_PSC" => ("Thrombolysis-capable center", COL[:h1]),
                "Route_NSC" => ("Non-stroke-center hospital", COL[:h2]))
    p = plot(size = sz(W1, 190), xlabel = "Onset-to-pickup time (min)",
             ylabel = "Extra road time to EVT-capable center\nover thrombolysis-capable center (min)",
             legend = :outerbottom, legend_columns = 2)
    for a in ("Route_CSC", "Route_PSC", "Route_NSC")
        g = tr[tr.Predicted_Action .== a, :]
        scatter!(p, g.Time_since_onset_min, g.Diff_CSC_PSC, color = tier[a][2], markersize = 2.2, markeralpha = 0.75,
                 label = "$(tier[a][1]) (n = $(nrow(g)))")
    end
    vline!(p, [TREE_SPLIT], color = :gray40, linestyle = :dot, linewidth = 0.6, label = "")
    hline!(p, [0], color = :gray40, linewidth = 0.6, label = "")
    savefig(p, joinpath(OUT, "fig_decision_space.pdf")); println("✓ fig_decision_space.pdf")
end

# ---------------------------------------------------------------------------
# Fig. dido: MDP gain as a function of door-in-door-out time (realized reward files)
# ---------------------------------------------------------------------------
function fig_dido()
    settings = [(60, "_dido60"), (90, "_dido90"), (121, ""), (175, "_dido175")]
    xs = Int[]; ms = Float64[]; hs = Float64[]
    for (dd, tag) in settings
        f = joinpath(SR, "CA_simulation_results_realized$(tag).csv")
        isfile(f) || (println("  missing $f, skipping"); continue)
        r = CSV.read(f, DataFrame)
        m, h = rep_ci(r.optimal_action_reward .- r.nearest_hospital_reward, r.replicate)
        push!(xs, dd); push!(ms, m); push!(hs, h)
    end
    p = plot(size = sz(W1, 160), xlabel = "Door-in-door-out time at sending hospital (min)",
             ylabel = "MDP gain over nearest-hospital\nrouting, P(mRS 0–1)", legend = false, ylims = (0, 0.05), xlims = (50, 185))
    vspan!(p, [89, 175], color = :gray92, linealpha = 0, label = "")
    annotate!(p, 132, 0.002, text("registry IQR", 6, :gray30))
    vline!(p, [121], color = :gray40, linestyle = :dot, linewidth = 0.6)
    plot!(p, xs, ms, yerror = hs, color = COL[:mdp], marker = :circle, markersize = 3)
    for (x, m) in zip(xs, ms); annotate!(p, x, m + 0.002, text(string(round(m, digits = 3)), 6, :gray30, :center)); end
    savefig(p, joinpath(OUT, "fig_dido.pdf")); println("✓ fig_dido.pdf")
end

# ---------------------------------------------------------------------------
# Fig. forest: robustness rows (noise, bias, EVT definition, onset cohort)
# ---------------------------------------------------------------------------
function fig_forest()
    rows = Tuple{String,String}[]   # (label, file)
    push!(rows, ("Base case", "CA_simulation_results_realized.csv"))
    for s in (10, 20); push!(rows, ("Travel-time noise σ = 0.$(s)", "CA_simulation_results_noise$(s).csv")); end
    for b in (70, 80, 90, 110, 120, 130); push!(rows, ("Travel-time bias ×$(b/100)", "CA_simulation_results_bias$(b).csv")); end
    push!(rows, ("EVT on-site definition", "CA_simulation_results_evtonsite.csv"))
    push!(rows, ("Uniform onset cohort", "CA_simulation_results_uniform.csv"))
    push!(rows, ("80 km first-leg catchment", "CA_simulation_results_catch80.csv"))
    labs = String[]; ms = Float64[]; hs = Float64[]
    for (lab, f) in rows
        path = joinpath(SR, f); isfile(path) || (println("  missing $f, skipping"); continue)
        r = CSV.read(path, DataFrame)
        m, h = rep_ci(r.optimal_action_reward .- r.nearest_hospital_reward, r.replicate)
        push!(labs, lab); push!(ms, m); push!(hs, h)
    end
    n = length(labs); y = collect(n:-1:1)
    p = plot(size = sz(W1, 40 + 14n), xlabel = "MDP gain over nearest-hospital routing",
             yticks = (y, labs), legend = false, ylims = (0.4, n + 0.6), grid = :x)
    vline!(p, [ms[1]], color = :gray40, linestyle = :dot, linewidth = 0.6)
    scatter!(p, ms, y, xerror = hs, color = COL[:mdp], markersize = 3)
    savefig(p, joinpath(OUT, "fig_sensitivity_forest.pdf")); println("✓ fig_sensitivity_forest.pdf")
end

# ---------------------------------------------------------------------------
# Fig. load: share of patients at the busiest hospitals, MDP vs nearest-EVT rule
# ---------------------------------------------------------------------------
shortname(h) = replace(String(h), "ROUTE_" => "", "MedicalCenter" => " MC", "Campus" => "", "Sutter" => " (Sutter)",
                        "KaiserFoundationHospital" => "Kaiser ", "HealthCare" => "", r"([a-z])([A-Z])" => s"\1 \2")
function fig_load()
    cm = countmap(d[!, ACT[:mdp]]); ch = countmap(d[!, ACT[:h1]])
    ksm = first.(sort(collect(cm), by = last, rev = true)); ksh = first.(sort(collect(ch), by = last, rev = true))
    top = unique(vcat(ksh[1:min(6, end)], ksm[1:min(8, end)]))[1:10]
    sm, sh = cm, ch
    n = length(top); y = collect(n:-1:1)
    a = [100 * get(sh, h, 0) / nrow(d) for h in top]; b = [100 * get(sm, h, 0) / nrow(d) for h in top]
    p = plot(size = sz(W1, 200), xlabel = "Share of patients sent to hospital (%)", yticks = (y, shortname.(top)),
             legend = :outerbottom, grid = :x, ylims = (0.4, n + 0.6))
    for i in 1:n; plot!(p, [a[i], b[i]], [y[i], y[i]], color = :gray75, linewidth = 1.2, label = ""); end
    scatter!(p, a, y, color = COL[:h1], markersize = 3.5, label = "$(LAB[:h1]) ($(length(sh)) hospitals used)")
    scatter!(p, b, y, color = COL[:mdp], markersize = 3.5, label = "$(LAB[:mdp]) ($(length(sm)) hospitals used)")
    savefig(p, joinpath(OUT, "fig_hospital_load.pdf")); println("✓ fig_hospital_load.pdf")
end

# ---------------------------------------------------------------------------
# Fig. map: binned mean paired difference on a lon/lat grid, with the Bay Area outline
# ---------------------------------------------------------------------------
function outline!(p)
    gj = JSON.parsefile("sampled_points/bay_area_outline.geojson")
    for f in gj["features"]
        g = f["geometry"]; polys = g["type"] == "MultiPolygon" ? g["coordinates"] : [g["coordinates"]]
        for poly in polys, ring in poly
            xs = [pt[1] for pt in ring]; ys = [pt[2] for pt in ring]
            plot!(p, xs, ys, color = :gray55, linewidth = 0.5, label = "")
        end
    end
end
function binned(lon, lat, v; step = 0.04, lonr = (-122.85, -121.4), latr = (37.2, 38.55), minn = 3)
    xs = lonr[1]:step:lonr[2]; ys = latr[1]:step:latr[2]
    S = zeros(length(ys), length(xs)); N = zeros(Int, length(ys), length(xs))
    for (x, y, z) in zip(lon, lat, v)
        i = clamp(floor(Int, (y - latr[1]) / step) + 1, 1, length(ys)); j = clamp(floor(Int, (x - lonr[1]) / step) + 1, 1, length(xs))
        S[i, j] += z; N[i, j] += 1
    end
    M = [N[i, j] >= minn ? S[i, j] / N[i, j] : NaN for i in 1:length(ys), j in 1:length(xs)]
    return xs, ys, M
end
function fig_map()
    src = isfile("sampled_points/CA_grid_patients.csv") ? "sampled_points/CA_grid_patients.csv" : joinpath(SR, "CA_simulation_results.csv")
    g = CSV.read(src, DataFrame); println("  map source: $src ($(nrow(g)) patients)")
    lon = hasproperty(g, :start_lon) ? g.start_lon : g.lon; lat = hasproperty(g, :start_lat) ? g.start_lat : g.lat
    panels = [(g.optimal_action_reward .- g.nearest_hospital_reward, "MDP minus nearest hospital", 0.06),
              (g.optimal_action_reward .- g.heuristic_1_reward, "MDP minus nearest EVT-capable center", 0.006)]
    ps = []
    for (v, ttl, vmax) in panels
        xs, ys, M = binned(lon, lat, v)
        p = heatmap(xs, ys, M, color = :RdBu_r, clims = (-vmax, vmax), aspect_ratio = 1 / cosd(37.8), title = ttl,
                    xlims = (-122.85, -121.4), ylims = (37.2, 38.55), xticks = false, yticks = false, grid = false,
                    colorbar_title = "Mean paired difference, P(mRS 0–1)", framestyle = :none)
        outline!(p)
        for (nm, x, y) in (("San Francisco", -122.42, 37.77), ("San Jose", -121.89, 37.34), ("Santa Rosa", -122.71, 38.44), ("Livermore", -121.77, 37.68))
            scatter!(p, [x], [y], color = :black, markersize = 2, label = ""); annotate!(p, x + 0.03, y + 0.02, text(nm, 6, :left))
        end
        push!(ps, p)
    end
    p = plot(ps..., layout = (1, 2), size = sz(W2, 240))
    savefig(p, joinpath(OUT, "fig_maps_binned.pdf")); println("✓ fig_maps_binned.pdf")
end

FIG in ("all", "onset")  && fig_onset()
FIG in ("all", "space")  && fig_space()
FIG in ("all", "dido")   && fig_dido()
FIG in ("all", "forest") && fig_forest()
FIG in ("all", "load")   && fig_load()
FIG in ("all", "map")    && fig_map()
println("done")
