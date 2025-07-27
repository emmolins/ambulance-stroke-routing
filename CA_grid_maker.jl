#=
File: grid_maker.jl
----------------------------------------------------------------------
Generates geospatial heatmaps of stroke patient outcome probabilities 
under different ambulance routing policies in the Bay Area. 
Compares an MDP-based optimal routing strategy to California's current 
policy, and visualizes improvement regions. Outputs include annotated 
PDF grid plots with masked coastlines and city labels.
=#

using GMT
using CSV
using DataFrames
using Statistics

# Create a heatmap for either the optimal or status quo routing policy
function make_plot_with_grid(option::String)
    # Define center of region and zoom level (centered at Hayward, CA)
    hayward_lat, hayward_lon = 37.6688, -122.0808
    zoom_level = 0.5
    region = [hayward_lon - zoom_level, hayward_lon + zoom_level,
              hayward_lat - zoom_level, hayward_lat + zoom_level]
    lon_min, lon_max, lat_min, lat_max = region

    # Set up a 200x200 grid of coordinates
    grid_size = 200
    lon_step = (lon_max - lon_min) / grid_size
    lat_step = (lat_max - lat_min) / grid_size
    X = [lon_min + i * lon_step for i in 0:grid_size]
    Y = [lat_min + j * lat_step for j in 0:grid_size]

    # Initialize outcome matrix
    colors = fill(NaN, grid_size, grid_size)

    # Loop through each grid cell and calculate mean probability of good outcome
    for i in 1:grid_size
        for j in 1:grid_size
            path = "sampled_points/grid_plot_csvs_CA/samples_data_$(i)_$(j).csv"
            if isfile(path)
                df = CSV.read(path, DataFrame)
                col = option == "smarter" ? :reward_best : :reward_CA
                colors[i, j] = mean(skipmissing(df[!, col]))
            end
        end
    end

    # Create color scale and apply landmask to exclude water regions
    cpt = makecpt(color=:hot, range=(0.1, 0.4, 0.001))
    colors_clean = map(x -> coalesce(x, NaN), colors)
    

    # Plot grid heatmap
    output_path = "simulation_results/$(option)_hotspot_gridplot.pdf"
    pcolor(X, Y, colors_clean',
           proj="M6i",
           cmap=cpt,
           region=region,
           frame="afg")
    coast!(region=region, proj="M6i", area=1000, water=:white)
    coast!(region=region, proj="M6i", shorelines=true)

    # Annotate cities on the map
    cities = [
        (-122.4194, 37.7749, "San Francisco"),
        (-121.8863, 37.3382, "San Jose"),
        (-122.2711, 37.8044, "Oakland"),
        (-121.9886, 37.5483, "Fremont"),
        (-121.9552, 37.3541, "Santa Clara"),
        (-122.0363, 37.3688, "Sunnyvale"),
        (-122.0808, 37.6688, "Hayward"),
        (-121.8058, 38.0049, "Antioch"),
        (-121.7680, 37.6819, "Livermore"),
        (-122.1430, 37.4419, "Palo Alto")
    ]

    # Only label cities inside the plotted region
    cities_in_region = filter(cities) do (lon, lat, _)
        lon_min <= lon <= lon_max && lat_min <= lat <= lat_max
    end

    if !isempty(cities_in_region)
        lons = [c[1] for c in cities_in_region]
        lats = [c[2] for c in cities_in_region]
        names = [c[3] for c in cities_in_region]

        # Plot city markers
        plot!(lons, lats,
              symbol="c0.15c",
              fill=:black,
              region=region,
              proj="M6i",
              show=false)

        # Label city names above dots
        for (lon, lat, name) in cities_in_region
            text!(name,
                  x=lon,
                  y=lat,
                  region=region,
                  proj="M6i",
                  font="8p,Times-Roman,black",
                  fill=:white,
                  pen="0.5p,black",
                  offset=(shift=(0, 0.3),),
                  show=false)
        end
    else
        println("Warning: No cities found within current region bounds.")
    end

    # Save final plot
    if option == "smarter"
        colorbar!(cmap=cpt, show=true, savefig=output_path)
    else
        coast!(show=true, savefig=output_path)
    end
    println("Plot saved to: $output_path")
end

# Create a heatmap showing the improvement of the optimal policy over the status quo
function make_difference_plot()
    # Define map region centered on Hayward
    hayward_lat, hayward_lon = 37.6688, -122.0808
    zoom_level = 0.5
    region = [hayward_lon - zoom_level, hayward_lon + zoom_level,
              hayward_lat - zoom_level, hayward_lat + zoom_level]
    lon_min, lon_max, lat_min, lat_max = region

    # Generate grid coordinates
    grid_size = 200
    lon_step = (lon_max - lon_min) / grid_size
    lat_step = (lat_max - lat_min) / grid_size
    X = [lon_min + i * lon_step for i in 0:grid_size]
    Y = [lat_min + j * lat_step for j in 0:grid_size]

    # Initialize difference grid
    diff_colors = fill(NaN, grid_size, grid_size)

    # Load outcome data and compute improvement (Optimal - CA)
    for i in 1:grid_size
        for j in 1:grid_size
            path = "sampled_points/grid_plot_csvs_CA/samples_data_$(i)_$(j).csv"
            if isfile(path)
                df = CSV.read(path, DataFrame; types=Dict(:reward_best => Float64, :reward_CA => Float64))
                best_vals = skipmissing(df[!, :reward_best])
                ca_vals = skipmissing(df[!, :reward_CA])

                if count(!ismissing, best_vals) >= 1 && count(!ismissing, ca_vals) >= 1
                    raw_diff = mean(best_vals) - mean(ca_vals)
                    diff_colors[i, j] = max(raw_diff, 0.0)  # Clip negative diffs to 0
                else
                    println("Missing data at ($i, $j)")
                    diff_colors[i, j] = NaN
                end
            end
        end
    end

    # Apply land mask
    cpt = makecpt(color=:turbo, range=(0.0, 0.15, 0.001))
    diff_colors_clean = map(x -> coalesce(x, NaN), diff_colors)

    # Plot difference heatmap
    output_path = "simulation_results/difference_policy_gridplot.pdf"
    pcolor(X, Y, diff_colors_clean',
           proj="M6i",
           cmap=cpt,
           region=region,
           frame="afg",
           title="Improvement in Probability (Optimal - Status Quo)")
    coast!(region=region, proj="M6i", area=1000, water=:white)
    coast!(region=region, proj="M6i", shorelines=true)

    # Add cities as markers and labels
    cities = [
        (-122.4194, 37.7749, "San Francisco"),
        (-121.8863, 37.3382, "San Jose"),
        (-122.2711, 37.8044, "Oakland"),
        (-121.9886, 37.5483, "Fremont"),
        (-121.9552, 37.3541, "Santa Clara"),
        (-122.0363, 37.3688, "Sunnyvale"),
        (-122.0808, 37.6688, "Hayward"),
        (-121.8058, 38.0049, "Antioch"),
        (-121.7680, 37.6819, "Livermore"),
        (-122.1430, 37.4419, "Palo Alto")
    ]

    cities_in_region = filter(cities) do (lon, lat, _)
        lon_min <= lon <= lon_max && lat_min <= lat <= lat_max
    end

    if !isempty(cities_in_region)
        lons = [c[1] for c in cities_in_region]
        lats = [c[2] for c in cities_in_region]
        names = [c[3] for c in cities_in_region]

        plot!(lons, lats,
              symbol="c0.15c",
              fill=:black,
              region=region,
              proj="M6i",
              show=false)

        for (lon, lat, name) in cities_in_region
            text!(name,
                  x=lon,
                  y=lat,
                  region=region,
                  proj="M6i",
                  font="8p,Times-Roman,black",
                  fill=:white,
                  pen="0.5p,black",
                  offset=(shift=(0, 0.3),),
                  show=false)
        end
    end

    colorbar!(cmap=cpt, show=true, savefig=output_path)
    println("Difference plot saved to: $output_path")
end

# Run all visualizations
println("Producing plot of p_good_outcome simulations with optimal policy")
make_plot_with_grid("smarter")

println("Producing plot for CA's current policy")
make_plot_with_grid("CA")

println("Producing difference plot (Optimal - Status Quo)")
make_difference_plot()
