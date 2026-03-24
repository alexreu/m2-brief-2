import pandas as pd


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)

    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }
    )

    return summary.sort_values("missing_pct", ascending=False)


def summarize_dataset(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_missing": int(df.isna().sum().sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
    }


def summarize_outliers(df: pd.DataFrame) -> dict:
    summary = {}
    num_df = df.select_dtypes(include="number")

    for column in num_df.columns:
        q1 = num_df[column].quantile(0.25)
        q3 = num_df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        if iqr == 0:
            summary[column] = {
                "count": 0,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
            continue

        outlier_mask = (num_df[column] < lower_bound) | (
            num_df[column] > upper_bound)

        summary[column] = {
            "count": int(outlier_mask.sum()),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return summary


def compare_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    before_summary = summarize_dataset(df_before)
    after_summary = summarize_dataset(df_after)

    return {
        "rows_removed": before_summary["rows"] - after_summary["rows"],
        "missing_reduced": before_summary["total_missing"] - after_summary["total_missing"],
        "before": before_summary,
        "after": after_summary,
    }


def print_cleaning_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    comparison = compare_before_after(df_before, df_after)

    print("\n====== Cleaning report ======\n")
    print(f"Before shape: {comparison['before']['shape']}\n")
    print(f"After shape: {comparison['after']['shape']}\n")
    print(f"Rows removed: {comparison['rows_removed']}\n")
    print(f"Missing values removed: {comparison['missing_reduced']}")

    print("\nMissing values by column before:")
    print(summarize_missing_values(df_before))

    print("\nMissing values by column after:")
    print(summarize_missing_values(df_after))

    print("\nOutliers after cleaning:")
    for column, stats in summarize_outliers(df_after).items():
        print(
            f"{column}: {stats['count']} outliers "
            f"(lower={stats['lower_bound']:.2f}, upper={stats['upper_bound']:.2f})"
        )
