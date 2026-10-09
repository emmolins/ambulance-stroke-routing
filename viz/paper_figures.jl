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
using Printf
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
    ytop = 0.0
    for (k, ls, lw) in ((:h2, :solid, 1.2), (:h1, :dash, 1.2), (:mdp, :solid, 2.0))
        diff = d[!, RW[k]] .- d[!, RW[:nh]]
        m = Float64[]; h = Float64[]
        for (a, b) in zip(edges[1:end-1], edges[2:end])
            sel = (d.t_onset .>= a) .& (d.t_onset .< b)
            mm, hh = rep_ci(diff[sel], d.replicate[sel]); push!(m, mm); push!(h, hh)
        end
        plot!(p, mids, m, ribbon = h, fillalpha = 0.15, color = COL[k], linestyle = ls, linewidth = lw,
              marker = :circle, markersize = 2.5, label = LAB[k])
        ytop = max(ytop, maximum(m .+ h))
    end
    ylims!(p, (-0.004, 1.45 * ytop))   # headroom keeps the legend clear of the curves
    vline!(p, [TREE_SPLIT], color = :gray40, linestyle = :dot, linewidth = 0.6, label = "")
    annotate!(p, TREE_SPLIT + 3, 0.5 * ytop, text("tree split\n$(Int(TREE_SPLIT)) min", 6, :gray30, :left))
    savefig(p, joinpath(OUT, "fig_gain_by_onset.pdf")); println("✓ fig_gain_by_onset.pdf")
end

# ---------------------------------------------------------------------------
# Fig. dist: distribution of per-patient outcome probability, nearest vs MDP
# (planning reward, all replicates), with means marked
# ---------------------------------------------------------------------------
function fig_dist()
    a = d[!, RW[:nh]]; b = d[!, RW[:mdp]]
    lo = floor(min(minimum(a), minimum(b)) / 0.01) * 0.01; hi = ceil(max(maximum(a), maximum(b)) / 0.01) * 0.01
    edges = lo:0.005:hi
    p = plot(size = sz(W1, 165), xlabel = "Probability of excellent outcome, P(mRS 0–1)", ylabel = "Patients",
             legend = :topleft, xlims = (lo, hi), left_margin = 3Plots.mm)
    histogram!(p, a, bins = edges, color = COL[:nh], fillalpha = 0.55, linecolor = :white, linewidth = 0.3,
               label = "Nearest hospital")
    histogram!(p, b, bins = edges, color = COL[:mdp], fillalpha = 0.55, linecolor = :white, linewidth = 0.3,
               label = "MDP policy")
    vline!(p, [mean(a)], color = COL[:nh], linestyle = :dash, linewidth = 1.2, label = "Mean, nearest hospital ($(round(mean(a), digits = 3)))")
    vline!(p, [mean(b)], color = COL[:mdp], linestyle = :dash, linewidth = 1.2, label = "Mean, MDP policy ($(round(mean(b), digits = 3)))")
    savefig(p, joinpath(OUT, "fig_outcome_dist.pdf")); println("✓ fig_outcome_dist.pdf")
end

# ---------------------------------------------------------------------------
# Fig. space: the MDP destination tier in the two dispatcher variables
# ---------------------------------------------------------------------------
function fig_space()
    tr = CSV.read("decision_tree_output/training_data_detailed.csv", DataFrame)
    tier = Dict("Route_CSC" => ("EVT-capable center", COL[:mdp]),
                "Route_PSC" => ("Thrombolysis-capable center", COL[:h1]),
                "Route_NSC" => ("Non-stroke center", COL[:h2]))
    ymax = maximum(tr.Diff_CSC_PSC)
    p = plot(size = sz(W1, 180), xlabel = "Onset-to-pickup time (min)",
             ylabel = "Extra road time to EVT-capable\ncenter vs. thrombolysis center (min)",
             legend = :topright, left_margin = 4Plots.mm, xlims = (30, 270), ylims = (-3, 1.4 * ymax))
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
const SHORT = Dict("ROUTE_JohnMuirMedicalCenterWalnutCreekCampus" => "John Muir, Walnut Creek",
                   "ROUTE_SutterEdenMedicalCenter"                => "Sutter Eden",
                   "ROUTE_CPMCDaviesCampusSutter"                 => "CPMC Davies",
                   "ROUTE_UCSFMedicalCenter"                      => "UCSF",
                   "ROUTE_MillsPeninsulaMedicalCenterSutter"      => "Mills-Peninsula",
                   "ROUTE_KaiserFoundationHospitalRedwoodCity"    => "Kaiser Redwood City",
                   "ROUTE_RegionalMedicalCenterOfSanJose"         => "Regional MC San Jose",
                   "ROUTE_ElCaminoHealthMountainView"             => "El Camino, Mountain View")
shortname(h) = get(SHORT, String(h), replace(String(h), "ROUTE_" => "", "MedicalCenter" => " MC", "Campus" => "",
                   "Sutter" => "", "KaiserFoundationHospital" => "Kaiser ", "HealthCare" => "", r"([a-z])([A-Z])" => s"\1 \2"))
function fig_load()
    cm = countmap(d[!, ACT[:mdp]]); ch = countmap(d[!, ACT[:h1]])
    ksm = first.(sort(collect(cm), by = last, rev = true)); ksh = first.(sort(collect(ch), by = last, rev = true))
    top = unique(vcat(ksh[1:min(6, end)], ksm[1:min(8, end)])); top = top[1:min(10, length(top))]
    sm, sh = cm, ch
    n = length(top); y = collect(n:-1:1)
    a = [100 * get(sh, h, 0) / nrow(d) for h in top]; b = [100 * get(sm, h, 0) / nrow(d) for h in top]
    p = plot(size = sz(W1, 185), xlabel = "Share of patients (%)", yticks = (y, shortname.(top)),
             legend = :outerbottom, legend_columns = 1, grid = :x, ylims = (0.4, n + 0.6), xlims = (0, 1.08 * max(maximum(a), maximum(b))),
             left_margin = 1Plots.mm, bottom_margin = 1Plots.mm)
    for i in 1:n; plot!(p, [a[i], b[i]], [y[i], y[i]], color = :gray75, linewidth = 1.2, label = ""); end
    scatter!(p, a, y, color = COL[:h1], markersize = 3.5, label = "Nearest EVT-capable center ($(length(sh)) hospitals)")
    scatter!(p, b, y, color = COL[:mdp], markersize = 3.5, label = "MDP policy ($(length(sm)) hospitals)")
    savefig(p, joinpath(OUT, "fig_hospital_load.pdf")); println("✓ fig_hospital_load.pdf")
end

# ---------------------------------------------------------------------------
# Fig. map: binned mean paired difference on a lon/lat grid, with the Bay Area outline
# ---------------------------------------------------------------------------
function outline!(p, file = "sampled_points/bay_area_outline.geojson")
    gj = JSON.parsefile(file)
    for f in gj["features"]
        g = f["geometry"]
        rings = g["type"] == "MultiPolygon" ? [ring for poly in g["coordinates"] for ring in poly] :
                g["type"] == "Polygon" ? g["coordinates"] : [g["coordinates"]]
        for ring in rings
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
        p = heatmap(xs, ys, M, color = cgrad(:RdBu, rev = true), clims = (-vmax, vmax), aspect_ratio = 1 / cosd(37.8), title = ttl,
                    xlims = (-122.85, -121.4), ylims = (37.2, 38.55), xticks = false, yticks = false, grid = false,
                    colorbar_title = "ΔP(mRS 0–1)", colorbar_titlefontsize = 7, colorbar_tickfontsize = 6,
                    right_margin = 3Plots.mm, framestyle = :none)
        outline!(p)
        for (nm, x, y) in (("San Francisco", -122.42, 37.77), ("San Jose", -121.89, 37.34), ("Santa Rosa", -122.71, 38.44), ("Livermore", -121.77, 37.68))
            scatter!(p, [x], [y], color = :black, markersize = 2, label = ""); annotate!(p, x + 0.03, y + 0.02, text(nm, 6, :left))
        end
        push!(ps, p)
    end
    p = plot(ps..., layout = (1, 2), size = sz(W2, 240))
    savefig(p, joinpath(OUT, "fig_maps_binned.pdf")); println("✓ fig_maps_binned.pdf")
end

# ---------------------------------------------------------------------------
# Figs. maps: per-cell outcome under two policies, and paired differences,
# drawn at the native grid resolution from sampled_points/CA_grid_cell_means.csv
# with a k x k neighbourhood mean (k = 5, about 3 km) to suppress the
# five-patient-per-cell sampling noise. Cell geometry comes from the grid mask.
# ---------------------------------------------------------------------------
const MASK_CSV  = "sampled_points/bay_area_grid_cells.csv"
const CELLS_CSV = "sampled_points/CA_grid_cell_means.csv"
# (name, lon, lat, side of the dot the label goes on)
const CITIES = [("San Francisco", -122.419, 37.775, :l), ("Oakland", -122.271, 37.804, :r), ("San Jose", -121.886, 37.338, :r),
                ("Fremont", -121.989, 37.548, :r), ("Palo Alto", -122.143, 37.442, :l), ("Livermore", -121.768, 37.682, :r),
                ("Antioch", -121.806, 38.005, :r), ("Concord", -122.031, 37.978, :l), ("Vallejo", -122.257, 38.104, :l),
                ("San Rafael", -122.531, 37.974, :l), ("Santa Rosa", -122.714, 38.440, :r), ("Napa", -122.286, 38.297, :r)]

function cell_geometry()
    mask = CSV.read(MASK_CSV, DataFrame)
    imin, imax = extrema(mask.cell_i); jmin, jmax = extrema(mask.cell_j)
    dlon = mask.lon_hi[1] - mask.lon_lo[1]; dlat = mask.lat_hi[1] - mask.lat_lo[1]
    lon0 = minimum(mask.lon_lo); lat0 = minimum(mask.lat_lo)
    X = lon0 .+ ((imin:imax) .- imin .+ 0.5) .* dlon
    Y = lat0 .+ ((jmin:jmax) .- jmin .+ 0.5) .* dlat
    return (imin = imin, jmin = jmin, X = collect(X), Y = collect(Y),
            ext = (lon0, lon0 + length(X) * dlon, lat0, lat0 + length(Y) * dlat))
end
function cell_matrix(cells::DataFrame, col::Symbol, g)
    M = fill(NaN, length(g.Y), length(g.X))
    for r in eachrow(cells); M[r.cell_j - g.jmin + 1, r.cell_i - g.imin + 1] = r[col]; end
    return M
end
function smooth_nan(M::Matrix{Float64}; k::Int = 5)
    r = k ÷ 2; nj, ni = size(M); out = fill(NaN, nj, ni)
    for j in 1:nj, i in 1:ni
        isnan(M[j, i]) && continue
        s = 0.0; c = 0
        for dj in -r:r, di in -r:r
            jj = j + dj; ii = i + di
            (1 <= jj <= nj && 1 <= ii <= ni && !isnan(M[jj, ii])) || continue
            s += M[jj, ii]; c += 1
        end
        out[j, i] = s / c
    end
    return out
end
degfmt(v, lat) = (d = floor(Int, abs(v)); m = round(Int, (abs(v) - d) * 60); m == 60 && (d += 1; m = 0);
                  @sprintf("%d°%02d'%s", d, m, lat ? (v >= 0 ? "N" : "S") : (v >= 0 ? "E" : "W")))
function map_panel(M, g, ttl; color, clims, colorbar, cbtitle)
    p = heatmap(g.X, g.Y, M, color = color, clims = clims, colorbar = colorbar, colorbar_title = cbtitle,
                colorbar_titlefontsize = 7, colorbar_tickfontsize = 6, title = ttl, titlefontsize = 8,
                xlims = (g.ext[1], g.ext[2]), ylims = (g.ext[3], g.ext[4]), aspect_ratio = 1 / cosd(38.0),
                xticks = -122.5:0.5:-121.5, yticks = 37.5:0.5:38.5, tickfontsize = 6,
                xformatter = v -> degfmt(v, false), yformatter = v -> degfmt(v, true),
                framestyle = :box, grid = true, gridalpha = 0.35, gridlinewidth = 0.3, left_margin = 1Plots.mm)
    outline!(p)
    for (nm, x, y, side) in CITIES
        scatter!(p, [x], [y], color = :black, markersize = 2, label = "")
        dx = side == :r ? 0.015 : -0.015
        w = 0.0165 * length(nm); x0 = side == :r ? x + dx : x + dx - w
        plot!(p, Shape([x0, x0 + w, x0 + w, x0], [y - 0.013, y - 0.013, y + 0.013, y + 0.013]),
              fillcolor = :white, fillalpha = 0.9, linecolor = :gray60, linewidth = 0.3, label = "")
        annotate!(p, x0 + w / 2, y, text(nm, 6, :center))
    end
    return p
end
function fig_maps()
    isfile(CELLS_CSV) || (println("  missing $CELLS_CSV, skipping maps"); return)
    cells = CSV.read(CELLS_CSV, DataFrame); g = cell_geometry()
    println("  map source: $CELLS_CSV ($(nrow(cells)) cells)")
    # (a) outcome under nearest-hospital routing and under the MDP, shared scale
    A = smooth_nan(cell_matrix(cells, :reward_nearest, g)); B = smooth_nan(cell_matrix(cells, :reward_optimal, g))
    v = filter(!isnan, vcat(vec(A), vec(B)))
    cl = (floor(quantile(v, 0.01) * 100) / 100, ceil(quantile(v, 0.99) * 100) / 100)
    p1 = map_panel(A, g, "Nearest-hospital routing"; color = cgrad(:YlOrRd), clims = cl, colorbar = false, cbtitle = "")
    p2 = map_panel(B, g, "MDP policy"; color = cgrad(:YlOrRd), clims = cl, colorbar = true, cbtitle = "P(mRS 0–1)")
    savefig(plot(p1, p2, layout = (1, 2), size = sz(W2, 250)), joinpath(OUT, "fig_maps_outcome.pdf")); println("✓ fig_maps_outcome.pdf")
    # (b) paired differences, each with its own symmetric scale
    ps = []
    for (col, ttl) in ((:diff_opt_minus_nearest, "MDP minus nearest hospital"), (:diff_opt_minus_heur1, "MDP minus nearest EVT-capable center"))
        D = smooth_nan(cell_matrix(cells, col, g)); m = ceil(quantile(abs.(filter(!isnan, vec(D))), 0.99) * 1000) / 1000
        push!(ps, map_panel(D, g, ttl; color = cgrad(:RdBu, rev = true), clims = (-m, m), colorbar = true, cbtitle = "ΔP(mRS 0–1)"))
    end
    savefig(plot(ps..., layout = (1, 2), size = sz(W2, 250)), joinpath(OUT, "fig_maps_diff.pdf")); println("✓ fig_maps_diff.pdf")
end

# ---------------------------------------------------------------------------
# Fig. policy: the MDP's first-destination tier from every cell at fixed onset
# times (scripts/policy_map.jl), with hospitals overlaid
# ---------------------------------------------------------------------------
const TIERCOL = Dict("CSC" => COL[:mdp], "PSC" => COL[:h1], "NSC" => COL[:h2])
function hospitals_ca()
    h = CSV.read("hospitals/CA_hospitals.csv", DataFrame)
    return [(String(r.Hospital), r.Lon, r.Lat, String(r.Type)) for r in eachrow(h)]
end
function fig_policy()
    f = "sampled_points/CA_policy_map.csv"
    isfile(f) || (println("  missing $f, skipping"); return)
    pm = CSV.read(f, DataFrame); g = cell_geometry(); hs = hospitals_ca()
    ts = sort(unique(pm.t_onset)); ps = []
    for t in ts
        sub = pm[pm.t_onset .== t, :]
        M = fill(NaN, length(g.Y), length(g.X))
        for r in eachrow(sub); M[r.cell_j - g.jmin + 1, r.cell_i - g.imin + 1] = r.tier == "CSC" ? 1.0 : r.tier == "PSC" ? 2.0 : 3.0; end
        p = heatmap(g.X, g.Y, M, color = cgrad([COL[:mdp], COL[:h1], COL[:h2]], categorical = true), clims = (0.5, 3.5),
                    colorbar = false, title = "Pickup $(Int(t)) min after onset", titlefontsize = 8, alpha = 0.75,
                    xlims = (g.ext[1], g.ext[2]), ylims = (g.ext[3], g.ext[4]), aspect_ratio = 1 / cosd(38.0),
                    xticks = -122.5:0.5:-121.5, yticks = 37.5:0.5:38.5, tickfontsize = 6,
                    xformatter = v -> degfmt(v, false), yformatter = v -> degfmt(v, true),
                    framestyle = :box, grid = true, gridalpha = 0.35, gridlinewidth = 0.3, left_margin = 1Plots.mm)
        outline!(p)
        for (nm, x, y, tier) in hs
            tier == "NSC" && continue
            scatter!(p, [x], [y], marker = tier == "CSC" ? :utriangle : :circle, markersize = tier == "CSC" ? 4.5 : 3,
                     color = :white, markerstrokecolor = :black, markerstrokewidth = 0.6, label = "")
        end
        push!(ps, p)
    end
    # legend panel row built from empty series
    leg = plot(framestyle = :none, legend = :top, legend_columns = 5, size = sz(W2, 20), legendfontsize = 7)
    for (lab, c) in (("to EVT-capable center", COL[:mdp]), ("to thrombolysis-capable center", COL[:h1]), ("to non-stroke center", COL[:h2]))
        plot!(leg, [NaN], [NaN], linewidth = 8, color = c, alpha = 0.75, label = lab)
    end
    scatter!(leg, [NaN], [NaN], marker = :utriangle, color = :white, markerstrokecolor = :black, label = "EVT-capable center")
    scatter!(leg, [NaN], [NaN], marker = :circle, color = :white, markerstrokecolor = :black, label = "thrombolysis-capable center")
    p = plot(plot(ps..., layout = (1, length(ps))), leg, layout = @layout([a{0.92h}; b{0.08h}]), size = sz(W2, 215))
    savefig(p, joinpath(OUT, "fig_policy_map.pdf")); println("✓ fig_policy_map.pdf")
end

# ---------------------------------------------------------------------------
# Fig. times: onset-to-needle (ischemic patients) and onset-to-puncture (LVO)
# under three policies (scripts/treatment_times.jl)
# ---------------------------------------------------------------------------
function fig_times()
    f = joinpath(SR, "CA_treatment_times.csv")
    isfile(f) || (println("  missing $f, skipping"); return)
    tt = CSV.read(f, DataFrame)
    pols = [("Nearest hospital", COL[:nh]), ("Nearest EVT-capable center", COL[:h1]), ("MDP policy", COL[:mdp])]
    edges = 0:15:480
    function panel(sel, col, ttl, xlab)
        p = plot(size = sz(W1, 150), xlabel = xlab, ylabel = "Share of patients", title = ttl, titlefontsize = 8,
                 legend = :topright, xlims = (0, 480))
        vline!(p, [270], color = :gray40, linestyle = :dash, linewidth = 0.7, label = "")
        for (name, c) in pols
            v = tt[(tt.policy .== name) .& sel, col]; v = filter(!isnan, v)
            isempty(v) && continue
            late = round(100 * mean(v .>= 270), digits = 0)
            stephist!(p, v, bins = edges, normalize = :probability, color = c, linewidth = 1.4,
                      label = "$name ($(Int(late))% after 270 min)")
        end
        return p
    end
    isch = (tt.stroke_type .== "LVO") .| (tt.stroke_type .== "NLVO")
    p1 = panel(isch, :t_needle, "Time to thrombolysis (ischemic stroke)", "Onset-to-needle time (min)")
    p2 = panel(tt.stroke_type .== "LVO", :t_puncture, "Time to thrombectomy (large-vessel occlusion)", "Onset-to-puncture time (min)")
    p = plot(p1, p2, layout = (1, 2), size = sz(W2, 160))
    savefig(p, joinpath(OUT, "fig_treatment_times.pdf")); println("✓ fig_treatment_times.pdf")
end

# ---------------------------------------------------------------------------
# Fig. ri: Rhode Island destinations under nearest-hospital routing and the MDP
# ---------------------------------------------------------------------------
function fig_ri()
    f = joinpath(SR, "RI_simulation_results.csv")
    isfile(f) || (println("  missing $f, skipping"); return)
    r = CSV.read(f, DataFrame); h = CSV.read("hospitals/RI_hospitals.csv", DataFrame)
    hname = Dict(String(x.Hospital) => String(x.DisplayName) for x in eachrow(h))
    hpos  = Dict(String(x.Hospital) => (x.Lon, x.Lat, String(x.Type)) for x in eachrow(h))
    pal = palette(:tab10)
    order = first.(sort(collect(countmap(r.optimal_action)), by = last, rev = true))
    colof = Dict(k => pal[mod1(i, 10)] for (i, k) in enumerate(order))
    ps = []
    for (col, ttl) in ((:nearest_hospital_action, "Nearest-hospital routing"), (:optimal_action, "MDP policy"))
        p = plot(size = sz(W1, 230), title = ttl, titlefontsize = 8, xlims = (-71.9, -71.1), ylims = (41.12, 42.04),
                 aspect_ratio = 1 / cosd(41.6), xticks = -71.8:0.4:-71.2, yticks = 41.2:0.4:42.0, tickfontsize = 6,
                 xformatter = v -> degfmt(v, false), yformatter = v -> degfmt(v, true), framestyle = :box,
                 grid = true, gridalpha = 0.35, gridlinewidth = 0.3, legend = :bottomleft, legendfontsize = 6)
        outline!(p, "sampled_points/ri_outline.geojson")
        cm = countmap(r[!, col])
        for k in order
            sel = r[!, col] .== k; sum(sel) == 0 && continue
            short = replace(k, "ROUTE_" => "")
            scatter!(p, r.start_lon[sel], r.start_lat[sel], color = colof[k], markersize = 1.1, markeralpha = 0.5,
                     label = "$(get(hname, short, short)) ($(round(100 * cm[k] / nrow(r), digits = 0) |> Int)%)")
        end
        for (nm, (x, y, tier)) in hpos
            scatter!(p, [x], [y], marker = tier == "CSC" ? :utriangle : tier == "PSC" ? :circle : :square,
                     markersize = tier == "CSC" ? 5 : 3.5, color = :white, markerstrokecolor = :black, markerstrokewidth = 0.7, label = "")
        end
        push!(ps, p)
    end
    p = plot(ps..., layout = (1, 2), size = sz(W2, 240))
    savefig(p, joinpath(OUT, "fig_ri_destinations.pdf")); println("✓ fig_ri_destinations.pdf")
end

FIG in ("all", "onset")  && fig_onset()
FIG in ("all", "dist")   && fig_dist()
FIG in ("all", "space")  && fig_space()
FIG in ("all", "dido")   && fig_dido()
FIG in ("all", "forest") && fig_forest()
FIG in ("all", "load")   && fig_load()
FIG in ("all", "map")    && fig_map()
FIG in ("all", "maps")   && fig_maps()
FIG in ("all", "policy") && fig_policy()
FIG in ("all", "times")  && fig_times()
FIG in ("all", "ri")     && fig_ri()
println("done")
