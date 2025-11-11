import pandas as pd

bech_movies = pd.read_parquet('datasets/output_data.parquet')
ratings  = pd.read_parquet('datasets/ratings.parquet')


print("ratings", len(ratings))


bech_movies = bech_movies.rename(columns={"id": "movieId"})
valid_ids = bech_movies["movieId"].dropna().unique()

filtered_rat = ratings[ratings["movieId"].isin(valid_ids)]


filtered_rat.to_parquet("ratings_filt.parquet", index=False)


print("rat", len(filtered_rat))


