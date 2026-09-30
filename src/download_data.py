import requests 
import time
import pandas as pd
dataset_id= "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

url = "https://data.gov.sg/api/action/datastore_search"
limit = 5000
offset = 0
all_records = []
while True:
    params = {
    "resource_id": dataset_id,
    "limit": limit,
    "offset": offset
    }
    response = requests.get(url,params = params)
    response.raise_for_status()
    data = response.json()
    records = data["result"]["records"]
    total = data["result"]["total"]
    all_records.extend(records)
    print(f"Downloaded {len(all_records)} / {total} records")
    offset+= len(records)
    if offset >= total:
        break
    time.sleep(2.6)
print("We are done boys")
print(f"Total downloads is {len(all_records)}")
df = pd.DataFrame(all_records)
df.to_csv("data/raw/hdb_resale.csv", index = False)
print(f"Saved to data/raw/hdb_resale.csv")