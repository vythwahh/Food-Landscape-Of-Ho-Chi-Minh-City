import requests
import json
import csv
import time

DISTRICTS = [
    ("Quan 1",      10.7769, 106.7009, 2500),
    ("Quan 2",      10.7873, 106.7519, 4000),
    ("Quan 3",      10.7797, 106.6886, 2500),
    ("Quan 4",      10.7583, 106.7047, 2500),
    ("Quan 5",      10.7553, 106.6618, 2500),
    ("Quan 6",      10.7478, 106.6346, 3000),
    ("Quan 7",      10.7324, 106.7218, 3500),
    ("Quan 8",      10.7232, 106.6283, 3500),
    ("Quan 9",      10.8437, 106.8006, 4000),
    ("Quan 10",     10.7736, 106.6680, 2500),
    ("Quan 11",     10.7632, 106.6513, 2500),
    ("Quan 12",     10.8682, 106.6432, 4000),
    ("Binh Thanh",  10.8122, 106.7099, 3500),
    ("Binh Tan",    10.7456, 106.6018, 5000),
    ("Go Vap",      10.8380, 106.6652, 3500),
    ("Thu Duc",     10.8700, 106.8030, 4500),
    ("Tan Binh",    10.8015, 106.6524, 3500),
    ("Tan Phu",     10.7908, 106.6276, 4500),
    ("Phu Nhuan",   10.7993, 106.6840, 2500),
]


def search_restaurants_osm(district_name, lat, lon, radius):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
[out:json][timeout:60];
(
  node["amenity"="restaurant"](around:{radius},{lat},{lon});
  node["amenity"="cafe"](around:{radius},{lat},{lon});
  node["amenity"="fast_food"](around:{radius},{lat},{lon});
);
out body;
"""
    response = requests.get(
        overpass_url,
        params={"data": query},
        headers={"User-Agent": "FoodscapeSaigon/1.0"},
        timeout=60
    )
    
    data = response.json()
    places = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        places.append({
            "district": district_name,
            "name": tags.get("name", "Unknown"),
            "cuisine": tags.get("cuisine", "unknown"),
            "amenity": tags.get("amenity", "unknown"),
            "lat": element.get("lat"),
            "lon": element.get("lon")
        })
    return places

all_places = []
for district, lat, lon, radius in DISTRICTS:
    print(f"Collecting {district} (radius={radius}m)...")
    try:
        places = search_restaurants_osm(district, lat, lon, radius)
        print(f"  → {len(places)} places")
        all_places.extend(places)
        time.sleep(10)
    except Exception as e:
        print(f"  → ERROR: {e}, skipping...")
        continue

with open("foodscape_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["district","name","cuisine","amenity","lat","lon"])
    writer.writeheader()
    writer.writerows(all_places)

print(f"\nDone! Total: {len(all_places)} places saved to foodscape_data.csv")