import pandas as pd

ratings = pd.read_csv("datasets/ratings.csv")
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

rat = ratings.merge(movies[["movieId", "imdbId"]], on="movieId", how="inner")
print(ratings.columns)
print(rat.columns)

rat_bechdel = rat.merge(bechdel, left_on="imdbId", right_on="imdbid", how="inner")
print(rat_bechdel.columns)

rat_bechdel = rat_bechdel.rename(columns={"rating_x": "rating"})
rat_bechdel = rat_bechdel[[
    "movieId", "imdbId", "rating"
]]

avg_ratings = rat_bechdel.groupby(["movieId", "imdbId"], as_index=False)["rating"].mean()
avg_ratings.to_parquet("datasets/avg_ratings_bechdel.parquet", index=False)

print(len(ratings))
print(len(avg_ratings))

