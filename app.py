import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# 1. Page Configuration
st.set_page_config(page_title="HCMC Food Landscape Dashboard", layout="wide")

st.title("Ho Chi Minh City Food Landscape Dashboard")
st.markdown("Spatial Analysis of Culinary Clusters using OpenStreetMap Data")
st.write("---")

# 2. Load the Cleaned Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_food_data.csv")
     
    df["cluster"] = df["cluster"].astype(str).str.strip()
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'cleaned_food_data.csv' not found. Please run main.py first!")
    st.stop()

# 3. Sidebar Filter Panel
st.sidebar.header("Interactive Filters")

 
cluster_display = {
    "All Clusters": "All Clusters",
    "0": "Outskirt / Low-density",
    "1": "Bustling residential",
    "2": "New urban / Young",
    "3": "Central affluent"
}

selected_cluster_label = st.sidebar.selectbox(
    "Select Spatial Cluster:", 
    options=list(cluster_display.values())
)

# Filter by Amenity Type
if "amenity" in df.columns:
    all_amenities = ["All Amenities"] + list(df["amenity"].dropna().unique())
    selected_amenity = st.sidebar.selectbox("Select Amenity Type:", all_amenities)
else:
    selected_amenity = "All Amenities"

# Apply Filters to DataFrame
filtered_df = df.copy()

if selected_cluster_label != "All Clusters":
     
    reverse_map = {v: k for k, v in cluster_display.items()}
    target_cluster = reverse_map[selected_cluster_label]
    
     
    filtered_df = filtered_df[
        (filtered_df["cluster"] == target_cluster) | 
        (filtered_df["cluster"].str.lower().str.contains(target_cluster.lower()))
    ]

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

hcmc_bounds = [[10.3, 106.3], [11.2, 107.1]]

m = folium.Map(
    location=[10.7769, 106.7009],
    zoom_start=12,
    min_zoom=10,
    max_zoom=18,
    scrollWheelZoom=False,
    max_bounds=True,
    bounds=hcmc_bounds,
    control_scale=True
)

 
color_palette = {
    '0': 'red', '1': 'blue', '2': 'green', '3': 'purple',
    'outskirt': 'red', 'bustling': 'blue', 'new urban': 'green', 'central': 'purple', 'central hub': 'purple'
}

# Add data points to the GIS map
for idx, row in filtered_df.iterrows():
    if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
        raw_cluster = str(row["cluster"]).lower().strip()
        
 
        color = "gray"
        display_name = "Chưa phân cụm"
        
        for key, val in color_palette.items():
            if key in raw_cluster:
                color = val
       
                if val == 'red': display_name = cluster_display["0"]
                elif val == 'blue': display_name = cluster_display["1"]
                elif val == 'green': display_name = cluster_display["2"]
                elif val == 'purple': display_name = cluster_display["3"]
                break
        
        place_name = str(row["name"]).replace("'", "\\'") if pd.notna(row["name"]) else "Unknown Outlet"
        cuisine_type = str(row["cuisine"]) if pd.notna(row["cuisine"]) else "Not Specified"
        amenity_type = str(row["amenity"]) if pd.notna(row["amenity"]) else "Not Specified"
        
        popup_html = f"""
        <div style='font-family: Arial, sans-serif; min-width: 160px;'>
            <h4 style='margin: 0 0 5px 0; color: #333;'>{place_name}</h4>
            <p style='margin: 3px 0;'><b>Type:</b> {amenity_type}</p>
            <p style='margin: 3px 0;'><b>Cuisine:</b> {cuisine_type}</p>
            <p style='margin: 3px 0; color: {color};'><b>Cluster:</b> {display_name}</p>
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
st_folium(m, use_container_width=True, height=650, returned_objects=[])