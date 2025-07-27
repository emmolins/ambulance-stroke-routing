# Ambulance Routing for Stroke Triage

This project simulates and analyzes intelligent ambulance routing strategies to optimize stroke patient outcomes. Two
geographic regions are analyzed: the San Francisco Bay Area and Rhode Island. For both regions, the status quo 
routing policy is compared with an MDP-based policy that incorporates treatment times, patient state, and geographic 
location. Visualizations and statistical analyses evaluate the outcome and travel-time tradeoffs of each strategy.

A decision tree–based policy is also constructed to approximate the MDP in an interpretable format, allowing clinical 
stakeholders to understand and potentially implement simplified versions of the optimal routing logic.

All code is written in Julia and organized by region (CA, RI) and function (modeling, simulation, analysis, and plotting).

## Project Structure

| File/Folder                  | Description |
|-----------------------------|-------------|
| `CA_STPMDP_ORS.jl`          | MDP model for California stroke triage with ORS travel times |
| `CA_simulations.jl`         | Runs patient simulations for California |
| `CA_simulations_stats.jl`   | Performs reward analysis, hypothesis testing, visualizations |
| `CA_grid_maker.jl`          | Generates geospatial heatmaps of outcome probabilities |
| `RI_STPMDP_ORS.jl`          | MDP model for Rhode Island (generalizability test) |
| `RI_simulations.jl`         | Runs simulations for RI |
| `RI_simulations_stats.jl`   | Analysis for RI model |
| `decision_tree_*.jl`        | Files for building and evaluating interpretable routing trees |
| `simulation_results/`       | Stores PDFs and output CSVs from simulation runs |
| `sampled_points/`           | Patient location data used for simulations and grid plotting |
| `hospitals/`                | Hospital metadata for routing decisions |
| `decision_tree_output/`     | Output of decision tree policy model |
| `gmt.history`               | Cache/history from GMT.jl plotting

## Key Features

- MDP-based stroke routing policy model
- Simulation of patient scenarios with geographic realism
- Comparative analysis of travel time and clinical outcome
- Interpretable decision tree approximations of routing logic
- Visual heatmaps of regional outcome improvements

## OpenRouteService (ORS) Setup

This project uses [OpenRouteService (ORS)](https://openrouteservice.org/) to estimate travel times between locations using
real road networks. The Julia code queries a **locally hosted ORS server** to compute realistic ambulance routing durations.

To run ORS locally, follow the instructions at https://giscience.github.io/openrouteservice/run-instance/

# Patient Sampling Methodology

Patient locations were sampled using high-resolution population density data from [Meta/CIESIN]
(https://data.humdata.org/dataset/united-states-high-resolution-population-density-maps-demographic-estimates) 
and filtered using a ZIP code shapefile from [Berkeley GeoData](https://geodata.lib.berkeley.edu/catalog/ark28722-s7888q). 
Sampling was done using Python with `rasterio`, `geopandas`, and `numpy`, and points were weighted by population density, 
filtered to fall on land, and exported to CSV for simulation input.

Code for sampling can be found at: https://colab.research.google.com/drive/1-EpK2cS_KADK8EofqnmQmmLS8yvy4fwK?usp=sharing

#  References
This project builds upon prior work in modeling stroke outcomes as a function of treatment timing:
Holodinsky, J. K., Williamson, T. S., Demchuk, A. M., Zhao, H., Zhu, L., Francis, M. J., ... & Kamal, N. (2018). Modeling
stroke patient outcomes based on time to endovascular treatment and other variables: the SPOT score. Stroke, 49(10), 2637–
2644. https://doi.org/10.1161/STROKEAHA.118.022792
