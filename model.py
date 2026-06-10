import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
import logging

 
# 0. CONFIGURATION & LOGGING SETUP  
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration - Automatically detects hardware acceleration (Step 2)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device for training: {DEVICE}")

 
# 1. DATA PROCESSING PIPELINE CLASS  
 
class FoodscapeDataPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.df = None
        self.district_df = None
        self.X_scaled = None

    def load_and_engineer_features(self):
        """Loads data and creates vectorized district-level spatial features."""
        logger.info(f"Loading raw dataset from {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        
        district_counts = self.df.groupby('district').size().rename('total_places')
        
        self.df['is_restaurant'] = (self.df['amenity'] == 'restaurant').astype(int)
        self.df['is_cafe'] = (self.df['amenity'] == 'cafe').astype(int)
        self.df['is_fastfood'] = (self.df['amenity'] == 'fast_food').astype(int)
        self.df['is_vietnamese'] = (self.df['cuisine'] == 'vietnamese').astype(int)
        self.df['is_coffee'] = self.df['cuisine'].str.contains('coffee', na=False).astype(int)
        self.df['is_international'] = self.df['cuisine'].isin(['burger', 'pizza', 'japanese', 'korean']).astype(int)
        self.df['is_unknown'] = (self.df['cuisine'] == 'unknown').astype(int)

        grouped = self.df.groupby('district').agg(
            restaurant_ratio=('is_restaurant', 'mean'),
            cafe_ratio=('is_cafe', 'mean'),
            fastfood_ratio=('is_fastfood', 'mean'),
            vietnamese_ratio=('is_vietnamese', 'mean'),
            coffee_ratio=('is_coffee', 'mean'),
            international_ratio=('is_international', 'mean'),
            unknown_cuisine_ratio=('is_unknown', 'mean'),
            cuisine_diversity=('cuisine', 'nunique')
        )
        
        self.district_df = district_counts.to_frame().join(grouped).reset_index()
        logger.info(f"Feature engineering completed. Shape: {self.district_df.shape}")
        return self.district_df

    def scale_features(self):
        """Standardizes features for the Deep Learning network."""
        feature_cols = [c for c in self.district_df.columns if c != 'district']
        X = self.district_df[feature_cols].values
        self.X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Feature normalization complete. Input dimension: {self.X_scaled.shape[1]}")
        return self.X_scaled

 
# 2. PYTORCH AUTOENCODER MODEL
 
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

 
# 3. MODEL TRAINER CLASS WITH DATALOADER  
 
class AutoencoderTrainer:
    def __init__(self, model, learning_rate=0.001, batch_size=8):
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size

    def fit(self, X_data, max_epochs=1000):
        tensor_x = torch.FloatTensor(X_data)
        dataset = TensorDataset(tensor_x, tensor_x)  # Self-reconstruction task
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info("Initiating model training loop via PyTorch DataLoader...")
        
        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(DEVICE)
                self.optimizer.zero_grad()
                reconstructed, _ = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
                
            total_epoch_loss = epoch_loss / len(X_data)
            
            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{max_epochs}] - Training Loss: {total_epoch_loss:.5f}")
                
        return self.model

 
# 4. MAIN RUNTIME EXECUTION
 
if __name__ == "__main__":
    # Data Processing Pipeline execution
    pipeline = FoodscapeDataPipeline('foodscape_data.csv')
    district_df = pipeline.load_and_engineer_features()
    X_scaled = pipeline.scale_features()
    
    print(district_df.set_index('district'))
    print(f"\nFeature matrix shape: {X_scaled.shape}")

    # Initialize Model and the new Trainer Class (Step 2)
    model = FoodscapeAutoencoder(input_dim=X_scaled.shape[1], embedding_dim=4)
    trainer = AutoencoderTrainer(model, learning_rate=0.001, batch_size=8)
    trained_model = trainer.fit(X_scaled, max_epochs=1000)

    # Extract latent embeddings safely using torch.no_grad()
    trained_model.eval()
    with torch.no_grad():
        full_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        _, embeddings_tensor = trained_model(full_tensor)
        embeddings_np = embeddings_tensor.cpu().numpy()

    print("\nEmbeddings shape:", embeddings_np.shape)

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(embeddings_np)

    district_df['cluster'] = clusters
    print("\nDistrict clusters:")
    print(district_df[['district', 'cluster']].sort_values('cluster'))

    # Visualise clusters on map
    colors = ['red', 'blue', 'green', 'purple']
    cluster_names = ['Mixed/Suburban', 'Outer Districts', 'Central Hub', 'Cafe District']

    m = folium.Map(location=[10.7769, 106.7009], zoom_start=12)

    df_raw = pipeline.df
    coords_cols = ['latitude', 'longitude'] if 'latitude' in df_raw.columns else ['lat', 'lon']
    district_coords = df_raw.groupby('district')[coords_cols].mean()

    for _, row in district_df.iterrows():
        district = row['district']
        cluster = int(row['cluster'])
        if district in district_coords.index:
            lat = district_coords.loc[district, coords_cols[0]]
            lon = district_coords.loc[district, coords_cols[1]]
            folium.CircleMarker(
                location=[lat, lon],
                radius=15,
                color=colors[cluster],
                fill=True,
                fill_opacity=0.7,
                popup=f"{district} — {cluster_names[cluster]}"
            ).add_to(m)

    m.save('cluster_map.html')
    print("Saved cluster_map.html")

    # Serialize model weights
    torch.save(trained_model.state_dict(), 'foodscape_model.pth')
    print("Model saved to foodscape_model.pth")