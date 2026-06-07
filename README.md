# HCMC-FoodScape — Food Landscape Spatial Analytics Engine

An enterprise-grade Geospatial Data Engineering and Spatial Analytics pipeline. HCMC-FoodScape processes large-scale location intelligence data from Ho Chi Minh City's culinary ecosystem, deploying advanced GIS algorithms and spatial clustering models to analyze vendor density, optimize commercial site selection, and uncover urban culinary distribution patterns.

---

## Geospatial Ingestion & Analytics Architecture

The core pipeline is engineered to ingest heterogeneous location data, convert it into geometric spatial data structures, and perform high-performance topological operations:

<pre>
+-------------------------+      +-------------------------+      +-------------------------+
| Raw Location Feeds      | ---> |  GeoPandas Data Pipeline| ---> | Topological Operations  |
| (OSM / Google Maps API) |      |   (CRS Transformation)  |      |   (Shapely Geometry)    |
+-------------------------+      +-------------------------+      +-------------------------+
                                                                               |
+-------------------------+      +-------------------------+                   v
|  Interactive Web Maps   | <--- | Spatial Clustering Engine| <--- +-------------------------+
| (Folium / Streamlit App)|      |  (DBSCAN / Spatial KDE) |      |   HCMC District Polygons|
+-------------------------+      +-------------------------+      |     (GeoJSON Layers)    |
                                                                  +-------------------------+
</pre>

---

## Spatial Core & Mathematical Foundation

Instead of treating coordinates as standard flat Euclidean points, the engine projectively transforms latitudes and longitudes into a local meter-based coordinate reference system (**WGS 84 / UTM Zone 48N - EPSG:32648**) to preserve true geographical distance and area metrics.

### 1. Spatial Density Metric (Kernel Density Estimation)
The localized culinary density $f(x)$ at any spatial vector coordinate $x$ is computed using a geographic Gaussian kernel:
$$f(x) = \frac{1}{n b^2} \sum_{i=1}^{n} K\left(\frac{d(x, x_i)}{b}\right)$$
Where:
- $d(x, x_i)$: The exact geodesic/Haversine distance between the target point and known food vendor $x_i$.
- $b$: The bandwidth radius (e.g., $500\text{m}$ buffer zone representing urban walking distance).

### 2. Topological Spatial Joins
Utilizes R-tree indexing natively via GeoPandas to execute high-speed Point-in-Polygon (PIP) operations:
$$\text{Is\_Within} = P_{\text{vendor}}(x, y) \cap \mathcal{M}_{\text{District\_Polygon}}$$

---

## Key Engineering Features

- **Advanced Coordinate Normalization:** Automatically handles Coordinate Reference System (CRS) transformations, safely re-projecting raw GPS points (`EPSG:4326`) to localized metric grids (`EPSG:32648`) for precise distance calculations.
- **Topological Overlay Analysis:** Employs `Shapely` to construct geographic buffer zones around transport hubs and universities, analyzing food landscape availability within a $500\text{m}$ radius.
- **Unsupervised Spatial Clustering:** Implements density-based spatial clustering algorithms to automatically detect high-concentration "food hotspots" while filtering out spatial noise.
- **High-Performance Geospatial Guardrails:** Seamlessly isolates heavy spatial cache indexing from compute tasks, allowing quick parsing of dense city-wide multi-polygon maps.

---

## Project Structure

<pre>
Food-Landscape-Of-Ho-Chi-Minh-City/
│
├── src/
│   ├── ingestion.py           # OSM/Google Maps API raw location data collector
│   ├── geospatial_core.py     # CRS projection, Shapely buffering, and Spatial Joins
│   ├── clustering.py          # Spatial density clustering & hotspot detection engine
│   └── visualization.py       # Folium & Streamlit interactive map generation
├── tests/
│   ├── test_geospatial.py     # Unit tests verifying CRS projections & coordinate limits
│   └── test_clustering.py     # Validation of spatial cluster output shapes
├── data/
│   └── hcmc_districts.geojson # Isolated geographic polygon boundaries of HCMC (Ignored if large)
├── requirements.txt           # Pinpointed geospatial Python dependencies
└── README.md                  # Comprehensive system and mathematical documentation
</pre>

---

## Execution & Visualization Guide

### 1. Environment Installation
Geospatial packages require specific binary matching. It is highly recommended to install via pip within a clean virtual environment:
```bash
git clone [https://github.com/vythwahh/Food-Landscape-Of-Ho-Chi-Minh-City-.git](https://github.com/vythwahh/Food-Landscape-Of-Ho-Chi-Minh-City-.git)
cd Food-Landscape-Of-Ho-Chi-Minh-City-
pip install -r requirements.txt
