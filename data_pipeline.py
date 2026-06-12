import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from shapely.geometry import Point, Polygon, MultiPolygon
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FoodscapeDataPipeline:
    def __init__(self, filepath, geojson_path=None):
        self.filepath = filepath
        self.geojson_path = geojson_path
        self.df = None
        self.districts_boundary = {}

        if geojson_path:
            self._load_geojson_boundaries()

    def _load_geojson_boundaries(self):
        """Loads district boundaries from a GeoJSON file using Shapely structures."""
        try:
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            for feature in geojson_data['features']:
                # Adjust key names depending on your specific GeoJSON structure (e.g., 'name', 'Ten_Quan')
                district_name = feature['properties'].get('name') or feature['properties'].get('Ten_Quan')
                geometry = feature['geometry']
                
                if geometry['type'] == 'Polygon':
                    poly = Polygon(geometry['coordinates'][0])
                elif geometry['type'] == 'MultiPolygon':
                    polys = [Polygon(coords[0]) for coords in geometry['coordinates']]
                    poly = MultiPolygon(polys)
                else:
                    continue
                
                if district_name:
                    self.districts_boundary[district_name] = poly
            logger.info(f"Loaded {len(self.districts_boundary)} district boundaries from GeoJSON.")
        except Exception as e:
            logger.error(f"Failed to parse GeoJSON boundaries: {str(e)}")

    def load_data(self):
        """Loads the raw dataset and standardizes basic coordinate names."""
        logger.info(f"Loading raw spatial dataset from {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        self.df = self.df.rename(columns={
            'Latitude': 'latitude', 'Longitude': 'longitude', 
            'lat': 'latitude', 'lon': 'longitude'
        })
        return self
    # 1 . ADVANCED DATA IMPUTATION
    def impute_missing_districts(self):
        """Predicts missing districts based on coordinates falling inside GeoJSON Polygons."""
        if not self.districts_boundary:
            logger.warning("Skipping district imputation: No GeoJSON boundaries loaded.")
            return self

        missing_mask = self.df['district'].isna() | (self.df['district'] == 'unknown')
        impute_count = 0

        for idx, row in self.df[missing_mask].iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                point = Point(row['longitude'], row['latitude']) # Shapely uses (X, Y) layout -> (lon, lat)
                for district_name, polygon in self.districts_boundary.items():
                    if polygon.contains(point):
                        self.df.at[idx, 'district'] = district_name
                        impute_count += 1
                        break
        
        logger.info(f"Spatial District Imputation complete. Resolved {impute_count} missing rows.")
        return self
    def impute_missing_cuisine_knn(self, k_neighbors=5, radius_meters=500):
        """Imputes missing cuisine entries via Spatial K-Nearest Neighbors using haversine metric within a radius."""
        # Convert 500m radius to radians approximate for BallTree earth radius conversion
        earth_radius_meters = 6371000.0
        radius_radians = radius_meters / earth_radius_meters

        # Mask rows where cuisine is known vs missing
        known_mask = self.df['cuisine'].notna() & (self.df['cuisine'] != 'unknown')
        missing_mask = self.df['cuisine'].isna() | (self.df['cuisine'] == 'unknown')

        if not missing_mask.any() or not known_mask.any():
            logger.info("No spatial cuisine imputation required or insufficient reference samples available.")
            return self

        # Extract radians for the spatial index
        known_coords = np.radians(self.df[known_mask][['latitude', 'longitude']].values)
        missing_coords = np.radians(self.df[missing_mask][['latitude', 'longitude']].values)

        # Build BallTree spatial indexing engine
        tree = BallTree(known_coords, metric='haversine')
        
        # Query matching candidates within the specified metric scope
        indices, distances = tree.query_radius(missing_coords, r=radius_radians, return_distance=True)
        
        known_cuisines = self.df[known_mask]['cuisine'].values
        impute_count = 0

        for missing_idx, (local_indices, local_distances) in enumerate(zip(indices, distances)):
            if len(local_indices) == 0:
                continue # No nearby places found within 500m, keep as unknown
            
            # Sort by distance and slice top K neighbors
            sorted_args = np.argsort(local_distances)[:k_neighbors]
            neighbor_indices = local_indices[sorted_args]
            neighbor_cuisines = known_cuisines[neighbor_indices]
            
            # Extract majority mode cuisine
            if len(neighbor_cuisines) > 0:
                majority_cuisine = pd.Series(neighbor_cuisines).mode()[0]
                actual_df_idx = self.df[missing_mask].index[missing_idx]
                self.df.at[actual_df_idx, 'cuisine'] = majority_cuisine
                impute_count += 1

        logger.info(f"Spatial Cuisine Imputation complete. Resolved {impute_count} missing cells using KNN.")
        return self
    def engineer_features(self):
        """Generates advanced spatial density, proximity, and regional diversity attributes."""
        logger.info("Initiating advanced spatial feature engineering pipeline...")
        coords_radians = np.radians(self.df[['latitude', 'longitude']].values)
        tree = BallTree(coords_radians, metric='haversine')
        earth_radius_meters = 6371000.0

        radius_1km_radians = 1000.0 / earth_radius_meters
        density_counts = tree.query_radius(coords_radians, r=radius_1km_radians, count_only=True)
        self.df['food_density_index'] = density_counts - 1

        notre_dame_rad = np.radians([10.7798, 106.6990])
        lat_diff = coords_radians[:, 0] - notre_dame_rad[0]
        lon_diff = coords_radians[:, 1] - notre_dame_rad[1]
        a = np.sin(lat_diff/2)**2 + np.cos(coords_radians[:, 0]) * np.cos(notre_dame_rad[0]) * np.sin(lon_diff/2)**2
        self.df['distance_to_center_hub_km'] = 2 * np.arcsin(np.sqrt(a)) * (earth_radius_meters / 1000.0)

        diversity_map = self.df.groupby('district')['cuisine'].nunique().to_dict()
        self.df['district_cuisine_diversity_score'] = self.df['district'].map(diversity_map).fillna(0)

        logger.info("Feature engineering phase executed successfully.")
        return self

    def save_pipeline_output(self, output_path):
        """Serializes processed dataset to designated CSV file destination."""
        self.df.to_csv(output_path, index=False)
        logger.info(f"Pipeline output successfully synchronized to {output_path}")
    # 2. FEATURE ENGINEERING
    
    def scale_features(self):
        """Standardizes features for the Deep Learning network."""
        from sklearn.preprocessing import StandardScaler
        
         
        feature_cols = [c for c in self.district_df.columns if c != 'district']
        X = self.district_df[feature_cols].values
        
         
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Feature normalization complete via Pipeline. Input dimension: {self.X_scaled.shape[1]}")
        return self.X_scaled

    def save_pipeline_output(self, output_path):
        """Serializes processed dataset to designated CSV file destination."""
        self.df.to_csv(output_path, index=False)
        logger.info(f"Pipeline output successfully synchronized to {output_path}")