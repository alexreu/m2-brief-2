import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules.analysis import analyse_dataset
from modules.cleaning import clean_dataset
from modules.reporting import print_cleaning_report


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

    df_before = df.copy()

    # Clean dataframe
    clean_df = clean_dataset(df)

    # Save cleaned dataframe
    clean_df.to_csv("./data/cleaned.csv", index=False)

    # Reporting
    print_cleaning_report(df_before, clean_df)


if __name__ == "__main__":
    main()
