import pandas as pd
import json
import time
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

DATA_DIR = Path("data")
MATCHES_FILE = DATA_DIR / "intl_results.csv"
COORDS_FILE = DATA_DIR / "country_coords.json"

def main():
    print("Loading intl_results.csv...")
    df = pd.read_csv(MATCHES_FILE)
    
    # We need coordinates for all home_teams, away_teams, and match countries
    countries = set(df['home_team']).union(set(df['away_team'])).union(set(df['country']))
    countries = {c for c in countries if isinstance(c, str)}
    print(f"Found {len(countries)} unique countries/teams to geocode.")
    
    coords = {}
    if COORDS_FILE.exists():
        with open(COORDS_FILE) as f:
            coords = json.load(f)
            
    geolocator = Nominatim(user_agent="world_cup_predictor")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)
    
    new_count = 0
    for country in list(countries):
        if country not in coords:
            print(f"Geocoding {country}...")
            try:
                location = geocode(country)
                if location:
                    coords[country] = {"lat": location.latitude, "lon": location.longitude}
                else:
                    print(f"  Warning: Could not geocode {country}")
                    coords[country] = {"lat": 0.0, "lon": 0.0}
            except Exception as e:
                print(f"Error on {country}: {e}")
                coords[country] = {"lat": 0.0, "lon": 0.0}
            
            new_count += 1
            
            # Save incrementally
            if new_count % 10 == 0:
                with open(COORDS_FILE, "w") as f:
                    json.dump(coords, f, indent=2)
                    
    with open(COORDS_FILE, "w") as f:
        json.dump(coords, f, indent=2)
        
    print(f"Done! Geocoded {new_count} new locations. Total coords: {len(coords)}")

if __name__ == "__main__":
    main()
