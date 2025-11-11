import pandas as pd

cast = pd.read_parquet("datasets/cast.parquet")
crew = pd.read_parquet("datasets/crew.parquet")
movies = pd.read_csv("datasets/links.csv")
bechdel = pd.read_parquet("datasets/output_data.parquet")



# 2 none imdbs
bechdel["imdbid"] = (
    bechdel["imdbid"]
    .astype(str)
    .str.strip()
    .replace("", None)  
)

bechdel["imdbid"] = pd.to_numeric(bechdel["imdbid"], errors="coerce").astype("Int64")
bechdel = bechdel.dropna(subset=["imdbid"]).reset_index(drop=True)

movies["imdbId"] = pd.to_numeric(movies["imdbId"], errors="coerce").astype("Int64")

cast = cast.merge(movies[["movieId", "imdbId"]], left_on="movie_id", right_on="movieId", how="inner")
crew = crew.merge(movies[["movieId", "imdbId"]], left_on="movie_id", right_on="movieId", how="inner")


cast_bechdel = cast.merge(bechdel, left_on="imdbId", right_on="imdbid", how="inner")
crew_bechdel = crew.merge(bechdel, left_on="imdbId", right_on="imdbid",how="inner")


cast_bechdel = cast_bechdel[[
    "movie_id", "person_id", "name", "character", "order", "gender", "imdbId", "title", "rating", "year"
]]

crew_bechdel = crew_bechdel[[
    "movie_id", "person_id", "name", "department", "job", "gender", "imdbId", "title", "rating", "year"
]]
# --- Save to Parquet ---
cast_bechdel.to_parquet("datasets/cast_bechdel.parquet", index=False)
crew_bechdel.to_parquet("datasets/crew_bechdel.parquet", index=False)

\

# --- 7️⃣ Sanity checks ---
print("Total cast rows:", len(cast))
print("Filtered cast rows:", len(cast_bechdel))
print("Total crew rows:", len(crew))
print("Filtered crew rows:", len(crew_bechdel))
print("Unique Bechdel-matched movies:", cast_bechdel['movie_id'].nunique())
print("\nSample merged cast rows:")

