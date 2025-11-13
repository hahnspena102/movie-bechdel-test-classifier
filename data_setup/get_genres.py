import pandas as pd
import ast


movies = pd.read_csv("datasets/movies_metadata.csv", low_memory=False)
bechdel = pd.read_parquet("datasets/output_data.parquet")


# --- Clean Bechdel IMDb IDs ---
bechdel["imdbid"] = (
    bechdel["imdbid"]
    .astype(str)
    .str.strip()
    .replace("", None)
    .str.zfill(7)
)
bechdel["imdbid"] = "tt" + bechdel["imdbid"]

# --- Clean movies_metadata IDs ---
movies = movies.rename(columns={"id": "movie_id"})
movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce").astype("Int64")
movies["imdb_id"] = movies["imdb_id"].astype(str).str.strip()


movies_bechdel = movies[movies["imdb_id"].isin(bechdel["imdbid"])].copy()


def parse_genres(x):
    try:
        genres_list = ast.literal_eval(x)
        if isinstance(genres_list, list):
            return [g["name"] for g in genres_list if "name" in g]
    except (ValueError, SyntaxError):
        return []
    return []

movies_bechdel["genres"] = movies_bechdel["genres"].apply(parse_genres)

# --- 5️⃣ Explode to one genre per row ---
genres_exploded = movies_bechdel.explode("genres").dropna(subset=["genres"])
genres_exploded = genres_exploded.rename(columns={"genres": "genre"})

# --- 6️⃣ Keep only relevant columns ---
genres_exploded = genres_exploded[["movie_id", "imdb_id", "title", "genre"]]

# --- 7️⃣ Save ---
genres_exploded.to_parquet("datasets/bechdel_movie_genres.parquet", index=False)

print("Movies in Bechdel list:", len(bechdel))
print("Movies matched in metadata:", len(movies_bechdel))
print("Genre rows created:", len(genres_exploded))
print(genres_exploded.head())
