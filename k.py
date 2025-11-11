import pandas as pd

bech_movies = pd.read_parquet('datasets/output_data.parquet')

print(len(bech_movies))