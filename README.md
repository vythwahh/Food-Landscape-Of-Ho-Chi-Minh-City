# Food Landscape of Ho Chi Minh City

A data-driven analysis of Saigon's food scene across 19 districts, combining OpenStreetMap data collection, exploratory analysis, and PyTorch-based district clustering.

## Overview

This project investigates the spatial distribution and digital representation of food establishments across HCMC. A key finding is that OSM data reflects **digital presence**, not actual restaurant density — revealing a measurable digital divide between central and peripheral districts.

## Project Structure
## 🔍 Key Findings

- **Central districts dominate** — Quan 1 and Quan 3 account for 35%+ of all mapped places, driven by digital-active customer bases
- **Digital inequality exists** — OSM coverage correlates with socioeconomic profile, not actual food density
- **Quan 7 outlier** — Lowest unknown cuisine ratio (~0.34) despite low total places, driven by Phu My Hung expat community
- **Restaurant > Cafe** — Despite perception, restaurants outnumber cafes 1.5:1 across the city
- **Quan 12 cafe anomaly** — Disproportionately high cafe ratio suggests a localized cluster

## Tech Stack

- **Data Collection:** Python, Overpass API (OpenStreetMap)
- **Analysis:** Pandas, Matplotlib, Seaborn
- **ML Model:** PyTorch (Autoencoder), Scikit-learn (KMeans)
- **Visualization:** Folium (interactive maps), Matplotlib
- **Deployment:** AWS Lambda + S3 *(in progress)*
- **Report:** LaTeX (Overleaf)

## Cluster Results

| Cluster | Districts | Characteristics |
|---|---|---|
| Central Hub | Q1, Q3, Q4, Binh Thanh, Phu Nhuan | High density, diverse cuisine |
| Urban Residential | Q5, Q10, Q11, Tan Binh, Go Vap | Mixed, moderate density |
| Sparse/Developing | Q2, Q6, Q7, Q8, Binh Tan | Low density, high international |
| Cafe Outlier | Quan 12 | Unusually high cafe ratio |

## Limitations

- OSM data reflects digital presence, not actual restaurant count
- Radius-based sampling may cause overlap between adjacent districts
- Thu Duc's student dining zone lies across the Binh Duong provincial border
- Future work: supplement with Foody/GrabFood data for ground truth comparison

## How to Run

```bash
# Install dependencies
pip install requests pandas matplotlib seaborn folium torch scikit-learn

# Collect data
python collect_data.py

# Run EDA
python eda.py

# Train model & cluster
python model.py

# Run analysis
python analysis.py
``` 