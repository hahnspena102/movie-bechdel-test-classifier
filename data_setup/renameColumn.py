import pandas as pd

# Read the parquet file
df = pd.read_parquet("./datasets/avg_ratings.parquet")

# Rename a column
df = df.rename(columns={"imdbId": "imdb_id"})

# Save it back to parquet
df.to_parquet("./datasets/avg.parquet", index=False)
