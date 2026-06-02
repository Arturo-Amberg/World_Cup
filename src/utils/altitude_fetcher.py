import pandas as pd
import requests
import time
import os

coords_file = 'data/country_coords.csv'
if not os.path.exists(coords_file):
    print("country_coords.csv not found!")
    exit()

df = pd.read_csv(coords_file)

if 'elevation' not in df.columns:
    df['elevation'] = 0.0

print(f"Fetching elevations for {len(df)} countries...")

# Open-Meteo provides free elevation data based on coordinates
for idx, row in df.iterrows():
    if row['elevation'] > 0:
        continue # Already fetched
        
    lat = row['Latitude']
    lon = row['Longitude']
    
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if 'elevation' in data and data['elevation']:
                elev = data['elevation'][0]
                df.at[idx, 'elevation'] = elev
                print(f"[{idx+1}/{len(df)}] {row['Country']}: {elev}m")
        time.sleep(0.1) # Be nice to the free API
    except Exception as e:
        print(f"Error fetching {row['Country']}: {e}")

df.to_csv(coords_file, index=False)
print("Elevations updated and saved.")
