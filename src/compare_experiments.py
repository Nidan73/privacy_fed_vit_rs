from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aggregate_results import collect_metric_rows


SUMMARY_COLUMNS = [
    "base_experiment_id",
    "mean_test_accuracy",
    "std_test_accuracy",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_weighted_f1",
    "std_weighted_f1",
    "number_of_runs",
]


def build_mean_std_summary(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    required_metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
    for column in required_metrics:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_metrics, how="all")
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    grouped = (
        df.groupby("base_experiment_id", dropna=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_macro_f1=("test_macro_f1", "mean"),
            std_macro_f1=("test_macro_f1", "std"),
            mean_weighted_f1=("test_weighted_f1", "mean"),
            std_weighted_f1=("test_weighted_f1", "std"),
            number_of_runs=("test_accuracy", "count"),
        )
        .reset_index()
    )
    return grouped[SUMMARY_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare repeated experiment runs by mean and standard deviation.")
    parser.add_argument("--metrics_dir", default="results/metrics", help="Directory containing metric JSON files.")
    parser.add_argument("--pattern", help="Optional glob pattern or substring for metric JSON files.")
    parser.add_argument(
        "--output_csv",
        default="results/tables/experiment_mean_std_summary.csv",
        help="Path for the mean/std comparison CSV.",
    )
    args = parser.parse_args()

    rows = collect_metric_rows(Path(args.metrics_dir), args.pattern)
    summary = build_mean_std_summary(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    print(f"Scanned metrics_dir: {args.metrics_dir}")
    print(f"Groups written: {len(summary)}")
    print(f"Output CSV: {output_csv}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
