import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyse_missing_values(df: pd.DataFrame):
    print(f"missing values :\n {df.isna().sum()}\n")

    percentage_missing_values_by_col = df.isna(
    ).mean().sort_values(ascending=False) * 100

    print(
        f"columns > 50% NaN :\n {percentage_missing_values_by_col[percentage_missing_values_by_col > 50]}\n")
    print(
        f"columns > 80% NaN :\n {percentage_missing_values_by_col[percentage_missing_values_by_col > 80]}")

    print(
        f"percentage of missing values by column :\n {percentage_missing_values_by_col.head(10)}")

    percentage_missing_values_by_row = df.isna().mean(
        axis=1).sort_values(ascending=False) * 100
    rows_with_missing_values = df.isna().any(axis=1).sum()
    print(f"rows with at least one NaN : {rows_with_missing_values}\n")
    print(
        f"top 10 rows by percentage of missing values :\n {percentage_missing_values_by_row.head(10)}\n")

    msno.matrix(df)
    plt.title("Missing Values Matrix")
    plt.show()

    percentage_missing_values_by_col.plot(kind="bar", figsize=(10, 7))
    plt.title("Percentage of Missing Values by Column")
    plt.ylabel("Percentage")
    plt.show()


def analyse_outliers(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")

    plt.figure(figsize=(20, 7))
    sns.boxplot(data=num_df, orient="v")
    plt.title("Global boxplot of the Data")
    plt.show()

    # IQR detection by column
    for column in num_df.columns:
        Q1 = num_df[column].quantile(0.25)
        Q3 = num_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if lower_bound == upper_bound:
            print(f"{column} -> 0 outliers (lower={lower_bound:.2f}, upper={upper_bound:.2f})")
            continue

        outlier_mask = (num_df[column] < lower_bound) | (
            num_df[column] > upper_bound)
        print(
            f"{column} -> {outlier_mask.sum()} outliers "
            f"(lower={lower_bound:.2f}, upper={upper_bound:.2f})"
        )


def compare_features(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")
    feature_pairs = [
        ("revenu_estime_mois", "montant_pret"),
        ("score_credit", "montant_pret"),
        ("age", "montant_pret"),
    ]

    for x_col, y_col in feature_pairs:
        if x_col in num_df.columns and y_col in num_df.columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=num_df, x=x_col, y=y_col, alpha=0.5)
            plt.title(f"{y_col} vs {x_col}")
            plt.show()


def plot_distribution(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")

    num_df.hist(figsize=(10, 7))
    plt.suptitle("Distribution of Numeric Variables")
    plt.tight_layout()
    plt.show()


def analyse_correlation(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")
    corr_matrix = num_df.corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation of numeric variables")
    plt.show()


def analyse_dataset(df: pd.DataFrame):
    print("====== Dataset analysis ======\n")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}\n")
    print(f"Summary statistics:\n{df.describe(include='all')}\n")

    analyse_missing_values(df)
    analyse_outliers(df)
    compare_features(df)
    plot_distribution(df)
    analyse_correlation(df)

    return df
