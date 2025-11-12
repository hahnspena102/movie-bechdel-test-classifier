import pandas as pd


csv_path = "./archive/movies_metadata.csv"              
output_parquet_path = "./datasets/movies_metadata.parquet"  
reference_parquet_path = "output_data.parquet"   

# === Load datasets ===
print("Loading files...")
movies_df = pd.read_csv(csv_path, low_memory=False)
reference_df = pd.read_parquet(reference_parquet_path)

# === Normalize column names ===
movies_df.columns = movies_df.columns.str.lower().str.strip()
reference_df.columns = reference_df.columns.str.lower().str.strip()

# === Rename columns if needed ===
if "imdbid" in reference_df.columns and "imdb_id" not in reference_df.columns:
    reference_df.rename(columns={"imdbid": "imdb_id"}, inplace=True)
if "imdbid" in movies_df.columns and "imdb_id" not in movies_df.columns:
    movies_df.rename(columns={"imdbid": "imdb_id"}, inplace=True)

# === Normalize IMDb ID formats ===
movies_df["imdb_id"] = movies_df["imdb_id"].astype(str).str.replace("^tt", "", regex=True).str.strip()
reference_df["imdb_id"] = reference_df["imdb_id"].astype(str).str.strip()

# === Filter by imdb_id ===
valid_imdb_ids = set(reference_df["imdb_id"].dropna().unique())
filtered_df = movies_df[movies_df["imdb_id"].isin(valid_imdb_ids)].copy()

# === Drop unnecessary columns ===
columns_to_drop = [
    "homepage",
    "belongs_to_collection",
    "adult",
    "genres",
    "poster_path",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "video",
    "status",
    
]


filtered_df.drop(columns=[c for c in columns_to_drop if c in filtered_df.columns], inplace=True)

# === Reorder columns ===
priority_cols = ["id", "imdb_id", "original_title", "title"]
existing_priority = [c for c in priority_cols if c in filtered_df.columns]
remaining_cols = [c for c in filtered_df.columns if c not in existing_priority]
filtered_df = filtered_df[existing_priority + remaining_cols]

# === Save to Parquet ===
filtered_df.to_parquet(output_parquet_path, index=False)
print(f"Saved {len(filtered_df)} movies to {output_parquet_path}")
print(f"Columns order: {existing_priority + remaining_cols}")
