import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium 
df = pd.read_csv('foodscape_data.csv')

print("=== BASIC INFO ===")
print(f"Total places: {len(df)}")
print(f"Districts: {df['district'].nunique()}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nAmenity types:\n{df['amenity'].value_counts()}")
print(f"\nTop 10 cuisines:\n{df['cuisine'].value_counts().head(10)}")

# plot 1: number of places by district
plt.figure(figsize=(12, 6))
district_counts = df.groupby('district').size().sort_values(ascending=False)
sns.barplot(x=district_counts.values, y=district_counts.index, palette='viridis')
plt.title('Number of Food Places by District (HCMC)')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('plot_district_counts.png', dpi=150)
plt.show()
print("Saved plot_district_counts.png")

# Plot 2: amenity type distribution
plt.figure(figsize=(8, 5))
df['amenity'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Food Place Types')
plt.xlabel('Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot_amenity_types.png', dpi=150)
plt.show()
print("Saved plot_amenity_types.png")
# Plot 3: cuisine by district (top cuisines only)
top_cuisines = df[df['cuisine'] != 'unknown']['cuisine'].value_counts().head(5).index
df_known = df[df['cuisine'].isin(top_cuisines)]

plt.figure(figsize=(14, 6))
cuisine_district = df_known.groupby(['district', 'cuisine']).size().unstack(fill_value=0)
cuisine_district.plot(kind='bar', figsize=(14, 6))
plt.title('Top Cuisines by District')
plt.xlabel('District')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plot_cuisine_district.png', dpi=150)
plt.show()
import folium

# Create a map centered on HCMC
m = folium.Map(location=[10.7769, 106.7009], zoom_start=12)

for _, row in df.iterrows():
    color = 'red' if row['amenity'] == 'restaurant' else 'blue' if row['amenity'] == 'cafe' else 'green'
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['name']} ({row['district']})"
    ).add_to(m)

m.save('foodscape_map.html')
print("Saved foodscape_map.html")
# Plot 4: International vs Vietnamese ratio by district
international_cuisines = ['burger', 'pizza', 'japanese', 'korean', 'chinese', 
                          'thai', 'indian', 'french', 'italian', 'american']

district_ratio = df.groupby('district').apply(lambda x: pd.Series({
    'vietnamese': (x['cuisine'] == 'vietnamese').sum() / len(x),
    'international': x['cuisine'].isin(international_cuisines).sum() / len(x),
    'other': (~x['cuisine'].isin(international_cuisines + ['vietnamese'])).sum() / len(x)
})).reset_index()

district_ratio = district_ratio.sort_values('international', ascending=False)

fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(district_ratio))
ax.bar(x, district_ratio['vietnamese'], label='Vietnamese', color='#e74c3c')
ax.bar(x, district_ratio['international'], bottom=district_ratio['vietnamese'], 
       label='International', color='#3498db')
ax.bar(x, district_ratio['other'], 
       bottom=district_ratio['vietnamese'] + district_ratio['international'],
       label='Other/Unknown', color='#95a5a6')

ax.set_xticks(list(x))
ax.set_xticklabels(district_ratio['district'], rotation=45, ha='right')
ax.set_title('Vietnamese vs International Cuisine Ratio by District')
ax.set_ylabel('Ratio')
ax.legend()
plt.tight_layout()
plt.savefig('plot_cuisine_ratio.png', dpi=150)
plt.show()
print("Saved plot_cuisine_ratio.png")