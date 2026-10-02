# ORS_SETUP — Local OpenRouteService server

The simulation scripts in this repo (`CA_STPMDP_ORS.jl`, `RI_STPMDP_ORS.jl`, `CA_simulations.jl`, `RI_simulations.jl`, `decision_tree_build.jl`) compute ambulance travel times by querying a **locally-hosted OpenRouteService (ORS)** server at:

```
http://localhost:8080/ors/v2/directions/driving-car
```

This document captures how to stand that server up on macOS so the Julia code can talk to it. The ORS server lives in its own folder (`~/ors`) outside the repo because it's an environment dependency, not project code.

## Prerequisites

- macOS with **Docker Desktop** installed and running (https://www.docker.com/products/docker-desktop/). After install, raise **Settings → Resources → Memory** to at least **8 GB** (12 GB recommended for the California graph build). Apply & Restart.
- ~6 GB of free disk space (1.2 GB OSM extract + 3–5 GB routing graph).

Verify Docker is up:

```sh
docker --version
docker compose version
```

## One-time setup

### 1. Create the ORS working directory

```sh
mkdir -p ~/ors/ors-docker/files
cd ~/ors
```

### 2. Download the OSM extract

The simulations cover California (primary region) and Rhode Island (generalizability test). California is the larger and more important of the two. Download from Geofabrik:

```sh
# California (~1.2 GB)
curl -L -o ors-docker/files/california-latest.osm.pbf \
  https://download.geofabrik.de/north-america/us/california-latest.osm.pbf

# Rhode Island (~30 MB) — only needed when running RI experiments
curl -L -o ors-docker/files/rhode-island-latest.osm.pbf \
  https://download.geofabrik.de/north-america/us/rhode-island-latest.osm.pbf
```

Verify the download isn't a truncated error page:

```sh
ls -lh ors-docker/files/
# california-latest.osm.pbf should be ~1.1–1.3 GB
```

### 3. Write `docker-compose.yml`

```sh
cat > ~/ors/docker-compose.yml << 'EOF'
services:
  ors-app:
    container_name: ors-app
    image: openrouteservice/openrouteservice:v8.0.0
    ports:
      - "8080:8082"          # host 8080 -> container 8082; Julia code expects 8080
    user: "1000:1000"
    volumes:
      - ./ors-docker:/home/ors
    environment:
      REBUILD_GRAPHS: "False"
      CONTAINER_LOG_LEVEL: "INFO"
      XMS: 1g
      XMX: 8g
      ors.engine.profile_default.build.source_file: /home/ors/files/california-latest.osm.pbf
      ors.engine.profiles.car.enabled: "true"
EOF
```

Notes:
- `8080:8082` — ORS v8 listens on 8082 inside the container; the Julia code is hardcoded to `localhost:8080`. The mapping bridges them without touching the code.
- `XMX: 8g` — Java heap ceiling. Bump to `12g` if Docker has more RAM allocated.
- To switch to Rhode Island, change `source_file` to `/home/ors/files/rhode-island-latest.osm.pbf` and restart.

### 4. Start ORS and let it build the graph

```sh
cd ~/ors
docker compose up
```

Leave the terminal open — logs stream live. The **first run**:
1. Pulls the ORS image (one-time, ~1 GB).
2. Builds the routing graph from the PBF. **20–40 minutes**, CPU-pegged. This is normal.

Look for `Started Application` (or similar "server ready") in the logs. That's when it's actually answering requests.

### 5. Verify

In a new terminal:

```sh
# Health check
curl http://localhost:8080/ors/v2/health

# Sample routing call (San Francisco → Oakland)
curl "http://localhost:8080/ors/v2/directions/driving-car?start=-122.4194,37.7749&end=-122.2712,37.8044"
```

A successful response is JSON with `features[0].properties.segments[0].duration` (seconds). That's exactly what the Julia code reads.

### Required: raise the route-distance limit

ORS caps routes at 100 km by default. Inter-facility transfers in the model are not distance-limited (reachability is decided by the road network), and several Bay Area hospital pairs exceed 100 km by road, so the cap must be raised or those calls fail with ORS error 2004. After the first start has written `ors-docker/config/ors-config.yml`, uncomment two lines under `ors: engine:` so the block reads

```yaml
    profile_default:
      maximum_distance: 400000
```

(leave the other `profile_default` keys commented), then `docker restart ors-app`. Verify with a long pair, e.g. Berkeley to Gilroy:

```sh
curl -s "http://localhost:8080/ors/v2/directions/driving-car?start=-122.25722,37.85547&end=-121.57178,37.03656" | head -c 80
```

A `FeatureCollection` means the limit took effect; an error with code 2004 means it did not. The Julia code treats a 2004 as unroutable rather than crashing, so a missed config change shows up as "distance limit exceeded" warnings in the run log.

## Day-to-day usage

```sh
cd ~/ors

# Start in background (graph is cached after first build; startup is now seconds)
docker compose up -d

# Stop
docker compose down

# Tail logs
docker compose logs -f ors-app
```

## Troubleshooting

- **First build fails with OOM** — raise Docker Desktop's memory and the `XMX` env var, then `docker compose down && docker compose up` (it'll resume from where it left off).
- **`docker compose up` complains the image was built from a different PBF** — set `REBUILD_GRAPHS: "True"` for one run, then back to `"False"`.
- **Julia code prints `Connection refused`** — ORS isn't running. `docker compose ps` from `~/ors` should show `ors-app` Up.
- **Julia code prints `404` warnings about non-routable points** — that's the code's handled case; it logs and returns `nothing`. Not an ORS problem.
- **Switching regions (CA ↔ RI)** — edit `source_file` in `docker-compose.yml`, set `REBUILD_GRAPHS: "True"` for the next start, run `docker compose up`. Once it's built the new graph, set back to `"False"`.
