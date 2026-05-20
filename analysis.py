import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('foodscape_data.csv')

# Add cluster info
clusters = {
    'Quan 1': 'Central Hub', 'Quan 3': 'Central Hub',
    'Quan 4': 'Central Hub', 'Binh Thanh': 'Central Hub',
    'Phu Nhuan': 'Central Hub',
    'Quan 10': 'Urban Residential', 'Go Vap': 'Urban Residential',
    'Tan Phu': 'Urban Residential', 'Tan Binh': 'Urban Residential',
    'Thu Duc': 'Urban Residential', 'Quan 9': 'Urban Residential',
    'Quan 5': 'Urban Residential', 'Quan 11': 'Urban Residential',
    'Quan 2': 'Sparse/Developing', 'Quan 6': 'Sparse/Developing',
    'Quan 7': 'Sparse/Developing', 'Quan 8': 'Sparse/Developing',
    'Binh Tan': 'Sparse/Developing',
    'Quan 12': 'Cafe Outlier'
}
df['cluster'] = df['district'].map(clusters)

# Plot 1: Digital divide — cuisine unknown ratio base on cluster
district_stats = df.groupby('district').agg(
    total=('name', 'count'),
    unknown_ratio=('cuisine', lambda x: (x=='unknown').mean()),
    cluster=('cluster', 'first')
).reset_index()

plt.figure(figsize=(12, 5))
colors_map = {'Central Hub': 'green', 'Urban Residential': 'blue',
              'Sparse/Developing': 'red', 'Cafe Outlier': 'purple'}
for cluster, group in district_stats.groupby('cluster'):
    plt.scatter(group['total'], group['unknown_ratio'],
                label=cluster, color=colors_map[cluster], s=100)
    for _, row in group.iterrows():
        plt.annotate(row['district'], (row['total'], row['unknown_ratio']),
                    textcoords="offset points", xytext=(5,5), fontsize=8)

plt.xlabel('Total Places (OSM coverage)')
plt.ylabel('Unknown Cuisine Ratio (data quality)')
plt.title('Digital Divide: OSM Coverage vs Data Quality by District')
plt.legend()
plt.tight_layout()
plt.savefig('plot_digital_divide.png', dpi=150)
plt.show()

# Plot 2: Cluster characterization
cluster_profile = df.groupby('cluster').agg(
    restaurant_ratio=('amenity', lambda x: (x=='restaurant').mean()),
    cafe_ratio=('amenity', lambda x: (x=='cafe').mean()),
    vietnamese_ratio=('cuisine', lambda x: (x=='vietnamese').mean()),
    international_ratio=('cuisine', lambda x: x.isin(['burger','pizza','japanese','korean']).mean()),
).reset_index()

cluster_profile.set_index('cluster').plot(kind='bar', figsize=(10, 5))
plt.title('Cluster Profiles — Food Characteristics')
plt.xlabel('Cluster')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('plot_cluster_profile.png', dpi=150)
plt.show()
print(cluster_profile.to_string())