import pandas as pd
import ast

# Load the credits CSV
df = pd.read_csv("datasets/credits.csv")

# Parse the stringified lists into real Python lists
def parse_list_column(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["cast"] = df["cast"].apply(parse_list_column)
df["crew"] = df["crew"].apply(parse_list_column)

# --- Expand CAST ---
cast_df = df.explode("cast").dropna(subset=["cast"]).reset_index(drop=True)
cast_expanded = pd.json_normalize(cast_df["cast"])

# Rename for clarity
cast_expanded = cast_expanded.rename(columns={"id": "person_id"})
cast_df = pd.concat(
    [cast_df[["id"]].rename(columns={"id": "movie_id"}).reset_index(drop=True),
     cast_expanded],
    axis=1
)

# --- Expand CREW ---
crew_df = df.explode("crew").dropna(subset=["crew"]).reset_index(drop=True)
crew_expanded = pd.json_normalize(crew_df["crew"])
crew_expanded = crew_expanded.rename(columns={"id": "person_id"})
crew_df = pd.concat(
    [crew_df[["id"]].rename(columns={"id": "movie_id"}).reset_index(drop=True),
     crew_expanded],
    axis=1
)

# --- (Optional) Keep only key columns ---
cast_df = cast_df[["movie_id", "person_id", "name", "character", "order", "gender"]]
crew_df = crew_df[["movie_id", "person_id", "name", "department", "job", "gender"]]

# --- Save the results ---
cast_df.to_parquet("datasets/cast.parquet", index=False)
crew_df.to_parquet("datasets/crew.parquet", index=False)

# Optional: also save to CSV for easy inspection
# cast_df.to_csv("cast.csv", index=False)
# crew_df.to_csv("crew.csv", index=False)
