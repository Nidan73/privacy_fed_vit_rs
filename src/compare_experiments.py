from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aggregate_results import collect_metric_rows


SUMMARY_COLUMNS = [
    "comparable_group_id",
    "base_experiment_id",
    "number_of_runs",
    "seeds_included",
    "mean_test_accuracy",
    "std_test_accuracy",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_weighted_f1",
    "std_weighted_f1",
]


def seed_is_missing(value: object) -> bool:
    return value is None or pd.isna(value)


def format_seeds(series: pd.Series) -> str:
    seeds: list[str] = []
    for value in series:
        if seed_is_missing(value):
            seeds.append("missing")
        else:
            seeds.append(str(int(value)))
    return ",".join(sorted(set(seeds)))


def build_mean_std_summary(rows: list[dict], include_missing_seed: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    if "seed" not in df.columns:
        df["seed"] = None

    missing_seed_mask = df["seed"].isna()
    missing_seed_rows = df[missing_seed_mask]
    if not include_missing_seed and not missing_seed_rows.empty:
        for experiment_id in missing_seed_rows["experiment_id"].tolist():
            print(f"Skipping missing-seed run from seed-stability summary: {experiment_id}")
        df = df[~missing_seed_mask]

    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    required_metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
    for column in required_metrics:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_metrics, how="all")
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    grouped = (
        df.groupby("comparable_group_id", dropna=False)
        .agg(
            base_experiment_id=("base_experiment_id", "first"),
            number_of_runs=("test_accuracy", "count"),
            seeds_included=("seed", format_seeds),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_macro_f1=("test_macro_f1", "mean"),
            std_macro_f1=("test_macro_f1", "std"),
            mean_weighted_f1=("test_weighted_f1", "mean"),
            std_weighted_f1=("test_weighted_f1", "std"),
        )
        .reset_index()
    )
    return grouped[SUMMARY_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare repeated experiment runs by mean and standard deviation.")
    parser.add_argument("--metrics_dir", default="results/metrics", help="Directory containing metric JSON files.")
    parser.add_argument("--pattern", help="Optional glob pattern or substring for metric JSON files.")
    parser.add_argument(
        "--include_missing_seed",
        action="store_true",
        help="Include runs without a recorded seed in the mean/std summary.",
    )
    parser.add_argument(
        "--output_csv",
        default="results/tables/experiment_mean_std_summary.csv",
        help="Path for the mean/std comparison CSV.",
    )
    args = parser.parse_args()

    rows = collect_metric_rows(Path(args.metrics_dir), args.pattern)
    summary = build_mean_std_summary(rows, include_missing_seed=args.include_missing_seed)
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
