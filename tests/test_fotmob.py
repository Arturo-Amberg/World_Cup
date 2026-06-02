from curl_cffi import requests
import json

r = requests.get("https://www.fotmob.com/api/searchapi/suggest?term=Copa+America", impersonate="chrome110")
print("Search:", r.status_code)
if r.status_code == 200:
    for item in r.json().get("suggest", {}):
        if item.get("type") == "league":
            print("Found league:", item.get("id"), item.get("name"))

r2 = requests.get("https://www.fotmob.com/api/searchapi/suggest?term=Nations+League", impersonate="chrome110")
if r2.status_code == 200:
    for item in r2.json().get("suggest", {}):
        if item.get("type") == "league":
            print("Found league:", item.get("id"), item.get("name"))
