import pandas as pd

from modules.analysis import analyse_dataset
from modules.cleaning import clean_dataset


def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()


def main():
    file_path = "./data/raw.csv"

    df = load_data(file_path)

    if df.empty:
        raise SystemExit("Dataset could not be loaded.")

    # Analyse du dataset avant nettoyage
    analyse_dataset(df)

    # Clean dataframe
    clean_df = clean_dataset(df)

    # Save cleaned dataframe
    clean_df.to_csv("./data/cleaned.csv", index=False)

    # Analyse du dataset après nettoyage
    analyse_dataset(clean_df)


if __name__ == "__main__":
    main()
