import requests
import csv
from io import StringIO

# Mapeo de códigos ELO a nombres de países — cubre los 48 clasificados al Mundial 2026
ELO_COUNTRY_CODES = {
    # Américas
    "AR": "Argentina",
    "BR": "Brazil",
    "CO": "Colombia",
    "UY": "Uruguay",
    "US": "USA",
    "MX": "Mexico",
    "CA": "Canada",
    "EC": "Ecuador",
    "PE": "Peru",
    "PY": "Paraguay",
    "CL": "Chile",
    "BO": "Bolivia",
    "VE": "Venezuela",
    "PA": "Panama",
    "CR": "Costa Rica",
    "HN": "Honduras",
    "JM": "Jamaica",
    "SV": "El Salvador",
    "TT": "Trinidad and Tobago",
    "GT": "Guatemala",
    # Europa
    "FR": "France",
    "EN": "England",
    "ES": "Spain",
    "PT": "Portugal",
    "NL": "Netherlands",
    "BE": "Belgium",
    "IT": "Italy",
    "DE": "Germany",
    "HR": "Croatia",
    "CH": "Switzerland",
    "TR": "Turkey",
    "PL": "Poland",
    "RS": "Serbia",
    "AT": "Austria",
    "DK": "Denmark",
    "SE": "Sweden",
    "NO": "Norway",
    "RO": "Romania",
    "UA": "Ukraine",
    "HU": "Hungary",
    "CZ": "Czech Republic",
    "SK": "Slovakia",
    "GR": "Greece",
    "AL": "Albania",
    "SI": "Slovenia",
    "GE": "Georgia",
    "ME": "Montenegro",
    "BA": "Bosnia and Herzegovina",
    "MK": "North Macedonia",
    "FI": "Finland",
    "IS": "Iceland",
    "IE": "Ireland",
    "SQ": "Scotland",
    "WA": "Wales",
    # África
    "MA": "Morocco",
    "SN": "Senegal",
    "NG": "Nigeria",
    "CM": "Cameroon",
    "GH": "Ghana",
    "CI": "Ivory Coast",
    "EG": "Egypt",
    "TN": "Tunisia",
    "DZ": "Algeria",
    "ML": "Mali",
    "ZA": "South Africa",
    "KE": "Kenya",
    "TZ": "Tanzania",
    "UG": "Uganda",
    "ZM": "Zambia",
    "MW": "Malawi",
    "BF": "Burkina Faso",
    "GN": "Guinea",
    # Asia
    "JP": "Japan",
    "KR": "South Korea",
    "AU": "Australia",
    "IR": "Iran",
    "SA": "Saudi Arabia",
    "QA": "Qatar",
    "IQ": "Iraq",
    "JO": "Jordan",
    "UZ": "Uzbekistan",
    "CN": "China",
    "TH": "Thailand",
    "VN": "Vietnam",
    "ID": "Indonesia",
    "OM": "Oman",
    "KW": "Kuwait",
    "BH": "Bahrain",
    "AE": "UAE",
    # Oceanía
    "NZ": "New Zealand",
    "FJ": "Fiji",
}

def fetch_elo_ratings():
    """
    Extrae los ratings ELO en vivo desde eloratings.net (TSV oficial).
    Retorna un diccionario { "CountryName": ELO }
    """
    print("Extrayendo ELO ratings desde eloratings.net...")
    url = "https://www.eloratings.net/World.tsv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        ratings = {}
        # TSV format is generally: Rank, Rank_diff, Code, Rating, ...
        # e.g.: 1   1   ES  2165    ...
        reader = csv.reader(StringIO(response.text), delimiter='\t')
        
        for row in reader:
            if len(row) >= 4:
                code = row[2]
                try:
                    rating = int(row[3])
                except ValueError:
                    continue
                
                # Asignar nombre si lo tenemos mapeado, si no, usar el código
                country_name = ELO_COUNTRY_CODES.get(code, code)
                ratings[country_name] = rating
                
        return ratings
    except Exception as e:
        print(f"Error extrayendo ratings ELO: {e}")
        return None

if __name__ == "__main__":
    elos = fetch_elo_ratings()
    if elos:
        print("Muestra de ELOs extraídos:")
        for country in ["Argentina", "Brazil", "Spain", "Mexico", "USA"]:
            print(f"{country}: {elos.get(country, 'No encontrado')}")
