# Food Landscape of Ho Chi Minh City (Saigon)

An advanced, data-driven spatial analysis pipeline of Saigon's food scene across 19 districts, combining spatial data imputation, automated data quality validation, PyTorch-based Deep Learning clustering, and an interactive GIS Dashboard.

## Overview & Key Insights

This project investigates the spatial distribution and digital representation of food establishments across HCMC. Key core findings include:
* The Digital Divide: OSM data reflects digital presence, not actual restaurant density, revealing a measurable digital inequality between central and peripheral districts.
* Central Hub Dominance: District 1 and District 3 account for a disproportionate volume of mapped places, driven by a highly digitally active consumer base.
* District 12 Anomaly: Separated into a unique low-density cluster with a distinct spatial footprint.
* New Urban Dynamics: Districts like Thu Duc and District 7 form a young, rapidly expanding cluster driven by modern urban projects and expat communities.

---

## Tech Stack

* Data Collection: Python, Overpass API (OpenStreetMap)
* Spatial Imputation & Engineering: Scikit-learn (K-Nearest Neighbors), Shapely, NumPy, Pandas
* Deep Learning Framework: PyTorch (Autoencoder for latent space embedding representation)
* Data Validation: Custom Automated Assertion Suite (DataQualityGuard)
* Interactive Visualization: Streamlit, Folium GIS Maps, Streamlit-Folium

---

## Project Architecture & Pipeline Flow

The system runs on a robust, synchronized vertical pipeline controlled via a central orchestrator:

1. data_pipeline.py (Spatial Engineering):
   * Executes Spatial District Imputation via Shapely polygon boundary containment.
   * Resolves missing categories using Spatial KNN Imputation (radius-based Haversine metric).
   * Engineers 3 advanced attributes: Food Density Index (1km radius), Proximity to Center Hub, and Regional Cuisine Diversity Score.
   
2. data_quality.py (Data Quality Guard):
   * Runs an automated compliance audit and enforces strict programmatic assertions.
   * Halts execution if thresholds fail, exporting a detailed data_quality_report.json.

3. model.py (PyTorch Layer):
   * Normalizes the 12-dimensional engineered matrix via a StandardScaler layer.
   * Trains a Spatial-Weighted Autoencoder to compress data into a 4D latent representation vector.
   * Executes KMeans to classify the latent representations into 4 distinct spatial clusters.

4. app.py (Interactive Dashboard):
   * A production-ready Streamlit interface allowing users to dynamically filter food outlets by cluster and amenity type on a customized Leaflet/Folium map.

---

## Deep Learning Cluster Results

| Cluster ID | Cluster Name | Representative Districts | Key Spatial Characteristics |
| :---: | :--- | :--- | :--- |
| 0 | Outskirt / Low-density | District 12 | Isolated peripheral zone, sparse food density, longer distance to center hub. |
| 1 | Bustling Residential | Binh Thanh, Q10, Phu Nhuan, Tan Binh, Q5, Q6, Q8, Q11, Binh Tan, Tan Phu, Q2 | Vibrant local dining scenes, high amenity counts, moderate-to-high urban density. |
| 2 | New Urban / Young | Thu Duc, District 7, Go Vap, District 9 | Developing megacity hubs, large student/expat populations, expanding international dining presence. |
| 3 | Central Affluent | District 1, District 3, District 4 | Core commercial hub, maximum culinary diversity, extremely high digital footprint and density index. |

---

## Limitations & Future Work

* OSM data is a proxy for digital activity, not absolute ground truth restaurant counts.
* Radius-based spatial engineering can capture boundary overlaps near adjacent districts.
* Future Extension: Supplement current dataset with API scraping from active food delivery platforms (GrabFood/ShopeeFood) for deep ground-truth validation.

---

## How to Run the System

### 1. Install Dependencies
``` 
pip install pandas numpy requests folium streamlit streamlit-folium shapely scikit-learn torch
```  

### 2. Execute the Central Processing & Training Pipeline
Run the central orchestrator to trigger raw data loading, spatial imputation, feature engineering, data quality testing, and PyTorch model training:
``` 
python main.py
```

### 3. Launch the Web Analytics Dashboard
Once main.py completes training and serializes the processed dataset, ignite the interactive Streamlit interface:
```
streamlit run app.py
 
```
