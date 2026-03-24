import pandas as pd
from sklearn.impute import KNNImputer


# Calculer le pourcentage de valeurs manquantes pour chaque ligne
def get_row_missing_percentage(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean(axis=1) * 100


# Supprimer les lignes les plus incompletes.
def drop_most_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_missing = get_row_missing_percentage(df)
    number_rows_to_drop = int(len(df) * 0.02)
    rows_to_drop = row_missing.sort_values(
        ascending=False).head(number_rows_to_drop).index
    return df.drop(index=rows_to_drop)


# Convertir en NaN les valeurs invalides et incoherentes
def replace_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(["?", "NA", "N/A", "null", "None"], pd.NA)
    if "loyer_mensuel" in df.columns:
        df.loc[df["loyer_mensuel"] < 0, "loyer_mensuel"] = pd.NA
    if "revenu_estime_mois" in df.columns:
        df.loc[df["revenu_estime_mois"] < 0, "revenu_estime_mois"] = pd.NA
    if "montant_pret" in df.columns:
        df.loc[df["montant_pret"] < 0, "montant_pret"] = pd.NA
    if "score_credit" in df.columns:
        df.loc[df["score_credit"] < 0, "score_credit"] = pd.NA
    return df


# Supprimer les colonnes sensibles peu utiles pour la suite du traitement
def drop_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ["poids", "taille"]
    existing_columns = [
        column for column in columns_to_drop if column in df.columns]
    return df.drop(columns=existing_columns)


# Detecter les outliers avec la methode IQR et retourner un resume par colonne
def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    outliers = {}
    num_df = df.select_dtypes(include="number")

    for column in num_df.columns:
        Q1 = num_df[column].quantile(0.25)
        Q3 = num_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if lower_bound == upper_bound:
            outliers[column] = {
                "count": 0,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
            continue

        outlier_mask = (num_df[column] < lower_bound) | (
            num_df[column] > upper_bound)
        outliers[column] = {
            "count": int(outlier_mask.sum()),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return outliers


# Ramener les valeurs extremes dans une zone acceptable sans supprimer les observations
def limit_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    num_df = df.select_dtypes(include="number")

    for column in columns:
        if column in num_df.columns:
            Q1 = num_df[column].quantile(0.25)
            Q3 = num_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df


# Imputer les valeurs manquantes restantes avec KNN
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    imputer = KNNImputer()
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


# Appliquer le pipeline complet de nettoyage
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = drop_sensitive_columns(df)
    df = replace_invalid_values(df)
    df = drop_most_incomplete_rows(df)
    outlier_report = detect_outliers_iqr(df)

    columns_with_outliers = [
        col for col, stats in outlier_report.items() if stats["count"] > 0]

    df = limit_outliers_iqr(df, columns_with_outliers)
    df = impute_missing_values(df)

    return df
