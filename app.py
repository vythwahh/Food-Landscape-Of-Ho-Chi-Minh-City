import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# 1. Page Configuration
st.set_page_config(page_title="HCMC Food Landscape Dashboard", layout="wide")

st.title("Ho Chi Minh City Food Landscape Dashboard")
st.markdown("### Spatial Analysis of Culinary Clusters using OpenStreetMap Data")
st.write("---")

# 2. Load the Cleaned Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_food_data.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'cleaned_food_data.csv' not found. Please run analysis.py first!")
    st.stop()

# 3. Sidebar Filter Panel
st.sidebar.header("Interactive Filters")

# Filter by Cluster Type
all_clusters = ["All Clusters"] + list(df["cluster"].dropna().unique())
selected_cluster = st.sidebar.selectbox("Select Spatial Cluster:", all_clusters)

# Filter by Amenity Type (e.g., restaurant, cafe) if available
if "amenity" in df.columns:
    all_amenities = ["All Amenities"] + list(df["amenity"].dropna().unique())
    selected_amenity = st.sidebar.selectbox("Select Amenity Type:", all_amenities)
else:
    selected_amenity = "All Amenities"

# Apply Filters to DataFrame
filtered_df = df.copy()

if selected_cluster != "All Clusters":
    filtered_df = filtered_df[filtered_df["cluster"] == selected_cluster]

if selected_amenity != "All Amenities":
    filtered_df = filtered_df[filtered_df["amenity"] == selected_amenity]

# 4. Top Metrics Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Food Outlets Displayed", value=len(filtered_df))
with col2:
    st.metric(label="Total Unique Districts", value=df["district"].nunique() if "district" in df.columns else "N/A")
with col3:
    st.metric(label="Identified Clusters", value=df["cluster"].nunique())

st.write("---")
# 5. Interactive GIS Map (Folium)
st.subheader("Interactive Geospatial Food Landscape")

# Define spatial boundaries for Ho Chi Minh City to avoid wild scrolling
hcmc_bounds = [[10.3, 106.3], [11.2, 107.1]]

# Initialize Folium Map with controlled zoom sensitivity
m = folium.Map(
    
    location=[10.7769, 106.7009],  # HCMC Center coordinates
    zoom_start=12,
    min_zoom=10,
    max_zoom=18,
    scrollWheelZoom=False,         # Disable scroll wheel zoom for smoother page scrolling
    max_bounds=True,               # Restrict map movement inside the bounding box
    bounds=hcmc_bounds,
    control_scale=True
)

# Define distinct color mapping for each cluster
color_map = {
    'Central Hub': 'green',
    'Urban Residential': 'blue',
    'Sparse/Developing': 'red',
    'Cafe Outlier': 'purple'
}

# Add data points to the GIS map
for idx, row in filtered_df.iterrows():
    if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
        cluster_name = row["cluster"]
        color = color_map.get(cluster_name, "gray")
        
        # Safe string handling for tooltips/popups
        place_name = str(row["name"]).replace("'", "\\'") if pd.notna(row["name"]) else "Unknown Outlet"
        cuisine_type = str(row["cuisine"]) if pd.notna(row["cuisine"]) else "Not Specified"
        amenity_type = str(row["amenity"]) if pd.notna(row["amenity"]) else "Not Specified"
        
        popup_html = f"""
        <div style='font-family: Arial, sans-serif; min-width: 150px;'>
            <h4 style='margin: 0 0 5px 0; color: #333;'>{place_name}</h4>
            <p style='margin: 3px 0;'><b>Type:</b> {amenity_type}</p>
            <p style='margin: 3px 0;'><b>Cuisine:</b> {cuisine_type}</p>
            <p style='margin: 3px 0; color: {color};'><b>Cluster:</b> {cluster_name}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

# Render Folium Map in Streamlit
st_folium(m, width=1200, height=650, returned_objects=[])

