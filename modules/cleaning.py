import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


EDUCATION_ORDER = {
    "aucun": 0,
    "bac": 1,
    "bac+2": 2,
    "bac+3": 3,
    "bac+4": 4,
    "master": 5,
    "doctorat": 6,
}

EDUCATION_ORDER_REVERSE = {value: key for key,
                           value in EDUCATION_ORDER.items()}


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
    columns_to_drop = ["nom", "prenom", "sexe",
                       "taille", "poids", "nationalité_francaise", "smoker"]
    existing_columns = [
        column for column in columns_to_drop if column in df.columns]
    return df.drop(columns=existing_columns)


# Detecter les outliers avec la methode IQR et retourner un resume par colonne
def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    outliers = {}
    num_df = df.select_dtypes(include="number")

    # On boucle sur chaque colonne numerique
    for column in num_df.columns:
        # On calcule le premier quartile de la colonne
        Q1 = num_df[column].quantile(0.25)
        # On calcule le troisieme quartile de la colonne
        Q3 = num_df[column].quantile(0.75)
        # On calcule l’ecart interquartile
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Si les deux bornes sont egales, il n'y a pas de dispersion exploitable
        if lower_bound == upper_bound:
            outliers[column] = {
                "count": 0,  # nombre outliers
                "lower_bound": float(lower_bound),  # borne basse
                "upper_bound": float(upper_bound),  # borne haute
            }
            continue

        # Cree un masque booleen : True pour les valeurs hors bornes
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


# On prepare certaines colonnes avant l'imputation
def preprocess_before_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # On transforme la date en variable numerique (plus adapte pour l'imputation)
    if "date_creation_compte" in df.columns:
        df["date_creation_compte"] = pd.to_datetime(
            # Si la conversion echoue, on remplace par NaT, equivalent a NaN pour les dates
            df["date_creation_compte"], errors="coerce"
        )
        # Date du jour
        ref_date = pd.Timestamp.today().normalize()
        # On calcule l'anciennete du compte en jours
        df["anciennete_compte_jours"] = (
            ref_date - df["date_creation_compte"]
        ).dt.days

        df = df.drop(columns=["date_creation_compte"])

    if "niveau_etude" in df.columns:
        df["niveau_etude"] = (df["niveau_etude"].astype(
            str).str.strip().str.lower())
        df["niveau_etude"] = df["niveau_etude"].replace(EDUCATION_ORDER)

    return df

# On ajuste certaines colonnes apres l'imputation pour avoir des valeurs cohérentes avec le métier


def postprocess_after_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "historique_credits" in df.columns:
        df["historique_credits"] = df["historique_credits"].round().clip(0, 5)

    # On remets les niveaux d'etudes en texte lisible
    if "niveau_etude" in df.columns:
        df["niveau_etude"] = df["niveau_etude"].round().clip(
            lower=min(EDUCATION_ORDER_REVERSE),
            upper=max(EDUCATION_ORDER_REVERSE),
        )
        df["niveau_etude"] = df["niveau_etude"].astype(int)
        df["niveau_etude"] = df["niveau_etude"].replace(
            EDUCATION_ORDER_REVERSE)

    # On arrondi l'ancienneté en jours et on interdis les valeurs negatives
    if "anciennete_compte_jours" in df.columns:
        df["anciennete_compte_jours"] = df["anciennete_compte_jours"].round().clip(
            lower=0
        )

    return df

# Imputer les valeurs manquantes en utilisant des strategies adaptees selon le type de variable
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_columns = [
        "sexe",
        "sport_licence",
        "region",
        "smoker",
        "nationalité_francaise",
        "situation_familiale",
    ]

    # On ne garde que les colonnes categorielles presentes dans le dataframe
    categorical_columns = [
        col for col in categorical_columns if col in df.columns]
    # On identifie les colonnes numeriques
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    # Imputation colonnes categorielles
    if categorical_columns:
        # On remplace les valeurs manquantes des colonnes categorielles par la modalite la plus frequente
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_columns] = cat_imputer.fit_transform(
            df[categorical_columns])

    # Imputation colonnes numeriques
    if numeric_columns:
        num_imputer = KNNImputer()
        df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])
    return df


# Appliquer le pipeline complet de nettoyage
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = replace_invalid_values(df)
    df = drop_most_incomplete_rows(df)
    df = preprocess_before_imputation(df)
    outlier_report = detect_outliers_iqr(df)

    columns_with_outliers = [
        col for col, stats in outlier_report.items() if stats["count"] > 0]

    df = limit_outliers_iqr(df, columns_with_outliers)
    df = impute_missing_values(df)
    df = postprocess_after_imputation(df)

    return df
