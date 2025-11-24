
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pathlib import Path

ROOT = Path(__file__).resolve().parent 
P = lambda name: ROOT / name


def read_and_split_data():
    df = pd.read_parquet('./datasets/movies_metadata.parquet')
    bechdel_df = pd.read_parquet('./datasets/bechdel_ratings.parquet')

    genres_df = pd.read_parquet('./datasets/genres.parquet')
    
    df = df.merge(bechdel_df[['imdb_id', 'bechdel_rating']], on='imdb_id', how='left')
    df['bechdel_pass'] = df['bechdel_rating'] >= 3
    
    genres_df_filtered = genres_df[genres_df['imdb_id'].isin(df['imdb_id'])]
    
    df_grouped = genres_df_filtered.groupby('imdb_id', as_index=False)['genre'].apply(list).rename(columns={'genre': 'all_genres'})

    df = df.merge(df_grouped, on='imdb_id', how='left')

    df['all_genres'] = df['all_genres'].fillna('[]') #some movies have no genres :(
    
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year

    m_train, m_test, _, _ = train_test_split(df, df, test_size=0.2, random_state=42)
    return m_train, m_test

if __name__ == "__main__":
    m_train, m_test = read_and_split_data()
    #print(len(m_train))
    #print(len(m_test))
