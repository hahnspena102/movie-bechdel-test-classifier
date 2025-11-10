import requests
import pandas as pd

    
api_url = "http://bechdeltest.com/api/v1/getAllMovies"
headers = {"Authorization": "Bearer YOUR_API_TOKEN"}
params = {"page": 1, "limit": 100}

response = requests.get(api_url, headers=headers, params=params)
response.raise_for_status() 
api_data = response.json() # Assuming the API returns JSON


df = pd.DataFrame(api_data)
output_filepath = "output_data.parquet"
df.to_parquet(output_filepath, index=False)