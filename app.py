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

