# prepare_data.py

import pandas as pd
from pathlib import Path

INPUT_CSV = "recipes_data.csv"       
CLEAN_PARQUET = "recipes_clean.parquet"
SAMPLE_CSV = "recipes_sample_clean.csv"

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
   
    possible_title_cols = ["title", "name", "recipe_name", "RecipeTitle"]
    possible_ingredients_cols = ["ingredients", "ingredient", "Ingredients"]
    possible_directions_cols = ["directions", "instructions", "steps", "method"]

    # Auto detect columns
    cols = df.columns.str.lower()
    def find(cols_try):
        for c in cols_try:
            if c in cols:
                return df.columns[[i for i,col in enumerate(df.columns) if col.lower()==c][0]]
        return None

    title_col = find(possible_title_cols)
    ing_col = find(possible_ingredients_cols)
    dir_col = find(possible_directions_cols)

    if title_col is None or ing_col is None or dir_col is None:
        raise ValueError("Could not detect title/ingredients/directions columns. Check the CSV and update column names in the script.")

    df = df.rename(columns={title_col: "title", ing_col: "ingredients", dir_col: "directions"})
    df = df[["title", "ingredients", "directions"]]

    # drop rows missing core fields
    df = df.dropna(subset=["title", "ingredients", "directions"])

    # strip whitespace
    df["title"] = df["title"].astype(str).str.strip()
    df["ingredients"] = df["ingredients"].astype(str).str.strip()
    df["directions"] = df["directions"].astype(str).str.strip()

    # drop duplicates (by title+ingredients)
    df = df.drop_duplicates(subset=["title", "ingredients"])

    return df

def main():
    p = Path(INPUT_CSV)
    if not p.exists():
        print(f"ERROR: {INPUT_CSV} not found. Put your Kaggle CSV in the project folder and name it {INPUT_CSV} or edit the variable.")
        return

    print("Loading CSV (this may be slow for very large files)...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print("Raw rows:", len(df))

    df_clean = clean_dataframe(df)
    print("Clean rows:", len(df_clean))

    df_clean.to_parquet(CLEAN_PARQUET, index=False)
    print("Saved cleaned data to:", CLEAN_PARQUET)

    sample_n = min(10000, len(df_clean))
    df_clean.sample(n=sample_n, random_state=42).to_csv(SAMPLE_CSV, index=False)
    print("Saved sample to:", SAMPLE_CSV)

if __name__ == "__main__":
    main()
