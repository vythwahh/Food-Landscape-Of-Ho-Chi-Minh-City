import logging
import os
import sys

 
from data_pipeline import FoodscapeDataPipeline
try:
    from data_quality import DataQualityGuard
except ImportError:
    DataQualityGuard = None

from model import FoodscapeAutoencoder, AutoencoderTrainer, DEVICE

import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_central_pipeline():
    logger.info("Trigger the central data processing and clustering pipeline for Saigon foodscape analysis...")

    csv_input = 'foodscape_data.csv'
    cleaned_output = 'cleaned_food_data.csv'

    # 1. Pipeline Execution: Load, Clean, Engineer, Cluster
    pipeline = FoodscapeDataPipeline(filepath=csv_input)
    pipeline.load_data()
    pipeline.impute_missing_districts()
    pipeline.impute_missing_cuisine_knn(k_neighbors=5, radius_meters=500)
    pipeline.engineer_features()

    # 2. Verify Data Quality: Run comprehensive checks and enforce strict assertions
    if DataQualityGuard:
        try:
            guard = DataQualityGuard(cleaned_output)
            guard.run_comprehensive_audit().enforce_strict_assertions()
            logger.info("Verify Data Quality: PASSED all checks. Ready for analysis.")
        except Exception as e:
            logger.warning(f"Alert data quality: {e}")

 
    logger.info("Normalizing features and training autoencoder for spatial clustering...")
    
 
    district_counts = pipeline.df.groupby('district').size().rename('total_places')
    pipeline.df['is_restaurant'] = (pipeline.df['amenity'] == 'restaurant').astype(int)
    pipeline.df['is_cafe'] = (pipeline.df['amenity'] == 'cafe').astype(int)
    pipeline.df['is_fastfood'] = (pipeline.df['amenity'] == 'fast_food').astype(int)
    pipeline.df['is_vietnamese'] = (pipeline.df['cuisine'] == 'vietnamese').astype(int)
    pipeline.df['is_coffee'] = pipeline.df['cuisine'].str.contains('coffee', na=False).astype(int)
    pipeline.df['is_international'] = pipeline.df['cuisine'].isin(['burger', 'pizza', 'japanese', 'korean']).astype(int)
    pipeline.df['is_unknown'] = (pipeline.df['cuisine'] == 'unknown').astype(int)

    grouped = pipeline.df.groupby('district').agg(
        restaurant_ratio=('is_restaurant', 'mean'),
        cafe_ratio=('is_cafe', 'mean'),
        fastfood_ratio=('is_fastfood', 'mean'),
        vietnamese_ratio=('is_vietnamese', 'mean'),
        coffee_ratio=('is_coffee', 'mean'),
        international_ratio=('is_international', 'mean'),
        unknown_cuisine_ratio=('is_unknown', 'mean'),
        cuisine_diversity=('cuisine', 'nunique'),
        avg_food_density=('food_density_index', 'mean'),
        avg_distance_to_center=('distance_to_center_hub_km', 'mean')
    )
    
    pipeline.district_df = district_counts.to_frame().join(grouped).reset_index()
    log_places = np.log1p(pipeline.district_df['total_places'])
    pipeline.district_df['data_confidence_score'] = log_places / (pipeline.district_df['unknown_cuisine_ratio'] + 1.0)
    
    X_scaled = pipeline.scale_features()

 
    model = FoodscapeAutoencoder(input_dim=X_scaled.shape[1], embedding_dim=4)
    trainer = AutoencoderTrainer(model, learning_rate=0.001, batch_size=8, patience=50)
    total_places_arr = pipeline.district_df['total_places'].values
    trained_model = trainer.fit(X_scaled, total_places_arr, max_epochs=1000)

    trained_model.eval()
    with torch.no_grad():
        full_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        _, embeddings_tensor = trained_model(full_tensor)
        embeddings_np = embeddings_tensor.cpu().numpy()
 
    kmeans = KMeans(n_clusters=4, random_state=42)
    district_clusters = kmeans.fit_predict(embeddings_np)
    
 
    district_to_cluster = dict(zip(pipeline.district_df['district'], district_clusters))
    
    
    pipeline.df['cluster'] = pipeline.df['district'].map(district_to_cluster).fillna(0).astype(int)

    
    pipeline.save_pipeline_output(cleaned_output)
    logger.info(f" SUCCESS. File saved at {cleaned_output}")

if __name__ == "__main__":
    run_central_pipeline()