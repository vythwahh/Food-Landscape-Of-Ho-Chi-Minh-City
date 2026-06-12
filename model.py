import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
import logging
from data_pipeline import FoodscapeDataPipeline
 
# CONFIGURATION & LOGGING SETUP
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration - Automatically detects hardware acceleration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device for training: {DEVICE}")

 
 # PYTORCH AUTOENCODER MODEL
 
class FoodscapeAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return reconstructed, embedding

 
# EARLY STOPPING MECHANISM
 
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-5):
        self.pvariance = patience  # using variable names subtly without tech-splaining
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

 
#  MODEL TRAINER CLASS WITH DATALOADER & EARLY STOPPING
 
class AutoencoderTrainer:
    def __init__(self, model, learning_rate=0.001, batch_size=8, patience=50):
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none') # Sửa ở đây để nhân trọng số từng hàng
        self.batch_size = batch_size
        self.early_stopping = EarlyStopping(patience=patience)

    def fit(self, X_data, total_places_arr, max_epochs=1000):
        tensor_x = torch.FloatTensor(X_data)
        
         
        weights = 1.0 / (total_places_arr + 1.0)
        weights = weights / weights.sum() * len(weights)
        tensor_weights = torch.FloatTensor(weights)
        
 
        dataset = TensorDataset(tensor_x, tensor_x, tensor_weights)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info("Initiating Spatial-Weighted model training loop via PyTorch DataLoader...")
        
        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_x, _, batch_w in dataloader:
                batch_x = batch_x.to(DEVICE)
                batch_w = batch_w.to(DEVICE)
                
                self.optimizer.zero_grad()
                reconstructed, _ = self.model(batch_x)
                
                 
                loss_elementwise = (reconstructed - batch_x) ** 2
                weighted_loss = loss_elementwise * batch_w.unsqueeze(1)
                loss = weighted_loss.mean()
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
                
            total_epoch_loss = epoch_loss / len(X_data)
            self.early_stopping(total_epoch_loss)
            
            if (epoch + 1) % 50 == 0 or self.early_stopping.early_stop:
                logger.info(f"Epoch [{epoch+1}/{max_epochs}] - Weighted Training Loss: {total_epoch_loss:.5f}")
                
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}. Halting training loop.")
                break
                
        return self.model

 
# MAIN RUNTIME EXECUTION
 
if __name__ == "__main__":
     
    pipeline = FoodscapeDataPipeline(filepath='foodscape_data.csv')
    pipeline.load_data()
    
     
    pipeline.impute_missing_districts()
    pipeline.impute_missing_cuisine_knn(k_neighbors=5, radius_meters=500)
    
     
    pipeline.engineer_features()
    
     
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
    logger.info(f"Feature matrix shape synced with advanced spatial features: {X_scaled.shape}")

     
    model = FoodscapeAutoencoder(input_dim=X_scaled.shape[1], embedding_dim=4)
    trainer = AutoencoderTrainer(model, learning_rate=0.001, batch_size=8, patience=50)
    
    total_places_arr = pipeline.district_df['total_places'].values
    trained_model = trainer.fit(X_scaled, total_places_arr, max_epochs=1000)

 
    trained_model.eval()
    with torch.no_grad():
        full_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        _, embeddings_tensor = trained_model(full_tensor)
        embeddings_np = embeddings_tensor.cpu().numpy()

    logger.info(f"Latent representations embeddings shape: {embeddings_np.shape}")

     
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(embeddings_np)
    pipeline.district_df['cluster'] = clusters
    
    logger.info(f"\nFinal Enhanced District Clusters Output:\n{pipeline.district_df[['district', 'cluster']].sort_values('cluster').to_string()}")

     
    colors = ['red', 'blue', 'green', 'purple']
    cluster_names = ['Mixed/Suburban', 'Outer Districts', 'Central Hub', 'Cafe District']

    m = folium.Map(location=[10.7769, 106.7009], zoom_start=12)
    df_raw = pipeline.df
    district_coords = df_raw.groupby('district')[['latitude', 'longitude']].mean()

    for _, row in pipeline.district_df.iterrows():
        district = row['district']
        cluster = int(row['cluster'])
        if district in district_coords.index:
            lat = district_coords.loc[district, 'latitude']
            lon = district_coords.loc[district, 'longitude']
            folium.CircleMarker(
                location=[lat, lon],
                radius=15,
                color=colors[cluster],
                fill=True,
                fill_opacity=0.7,
                popup=f"<b>{district}</b><br>Cluster: {cluster_names[cluster]}"
            ).add_to(m)

    m.save('cluster_map.html')
    logger.info("Saved enhanced cluster_map.html successfully.")

 
    torch.save(trained_model.state_dict(), 'foodscape_model.pth')
    logger.info("Model state weights serialized to foodscape_model.pth successfully!")