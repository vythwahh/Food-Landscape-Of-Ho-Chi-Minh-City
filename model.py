import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium

df = pd.read_csv('foodscape_data.csv')

# Tạo feature matrix cho mỗi quận
def create_district_features(df):
    features = []
    districts = df['district'].unique()
    
    for district in districts:
        d = df[df['district'] == district]
        total = len(d)
        
        feat = {
            'district': district,
            'total_places': total,
            'restaurant_ratio': len(d[d['amenity']=='restaurant']) / total,
            'cafe_ratio': len(d[d['amenity']=='cafe']) / total,
            'fastfood_ratio': len(d[d['amenity']=='fast_food']) / total,
            'vietnamese_ratio': len(d[d['cuisine']=='vietnamese']) / total,
            'coffee_ratio': len(d[d['cuisine'].str.contains('coffee', na=False)]) / total,
            'international_ratio': len(d[d['cuisine'].isin(['burger','pizza','japanese','korean'])]) / total,
            'unknown_cuisine_ratio': len(d[d['cuisine']=='unknown']) / total,
            'cuisine_diversity': d['cuisine'].nunique(),
        }
        features.append(feat)
    
    return pd.DataFrame(features)

district_df = create_district_features(df)
print(district_df.set_index('district'))

# Normalize features
feature_cols = [c for c in district_df.columns if c != 'district']
X = district_df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeature matrix shape: {X_scaled.shape}")
# Pytorch 
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

# Train
X_tensor = torch.FloatTensor(X_scaled)
model = FoodscapeAutoencoder(input_dim=X_scaled.shape[1], embedding_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    reconstructed, embedding = model(X_tensor)
    loss = criterion(reconstructed, X_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.4f}")

# Get embeddings
with torch.no_grad():
    _, embeddings = model(X_tensor)
    embeddings_np = embeddings.numpy()

print("\nEmbeddings shape:", embeddings_np.shape)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(embeddings_np)

district_df['cluster'] = clusters
print("\nDistrict clusters:")
print(district_df[['district', 'cluster']].sort_values('cluster'))
# Visualise clusters on map
import folium

colors = ['red', 'blue', 'green', 'purple']
cluster_names = ['Mixed/Suburban', 'Outer Districts', 'Central Hub', 'Cafe District']

m = folium.Map(location=[10.7769, 106.7009], zoom_start=12)

district_coords = df.groupby('district')[['lat','lon']].mean()

for _, row in district_df.iterrows():
    district = row['district']
    cluster = int(row['cluster'])
    if district in district_coords.index:
        lat = district_coords.loc[district, 'lat']
        lon = district_coords.loc[district, 'lon']
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
# Save model
torch.save(model.state_dict(), 'foodscape_model.pth')
print("Model saved to foodscape_model.pth")