from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, PowerNorm


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
SPLITS_DIR = ROOT / "data" / "splits"
RAW_DIR = ROOT / "data" / "raw"
FIGURE_DIR = ROOT / "papers" / "draft" / "figures"

PNG_DPI = 300


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    slug: str
    confusion_csv: Path
    primary: bool = False


EXPERIMENTS = [
    ExperimentSpec(
        experiment_id="N4D01HN",
        slug="n4d01hn",
        confusion_csv=METRICS_DIR
        / "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_confusion_matrix.csv",
        primary=True,
    ),
    ExperimentSpec(
        experiment_id="N2D01",
        slug="n2d01",
        confusion_csv=METRICS_DIR
        / "N2D01_fedavg_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_confusion_matrix.csv",
    ),
]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)


SHORT_LABELS = {
    "baseball_diamond": "baseball",
    "basketball_court": "basketball",
    "circular_farmland": "circular farm",
    "commercial_area": "commercial",
    "dense_residential": "dense res.",
    "golf_course": "golf",
    "ground_track_field": "track field",
    "industrial_area": "industrial",
    "medium_residential": "medium res.",
    "mobile_home_park": "mobile homes",
    "parking_lot": "parking lot",
    "railway_station": "rail station",
    "rectangular_farmland": "rect. farm",
    "sparse_residential": "sparse res.",
    "storage_tank": "storage tank",
    "tennis_court": "tennis",
    "thermal_power_station": "thermal plant",
}


generated_files: list[Path] = []
skipped: list[tuple[str, str]] = []
sources_used: list[Path] = []


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.05)
        generated_files.append(path)
        print(f"generated: {path}")
    plt.close(fig)


def skip(stem: str, reason: str) -> None:
    skipped.append((stem, reason))
    print(f"skipped: {stem} ({reason})")


def short_label(class_name: str) -> str:
    return SHORT_LABELS.get(class_name, class_name.replace("_", " "))


def display_pair(true_class: str, predicted_class: str) -> str:
    return f"{short_label(true_class)} -> {short_label(predicted_class)}"


def load_class_order(confusion_df: pd.DataFrame) -> list[str]:
    mapping_path = RAW_DIR / "nwpu_resisc45" / "class_mapping.csv"
    split_path = SPLITS_DIR / "nwpu" / "train.csv"

    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        if {"label", "class_name"}.issubset(mapping.columns):
            return mapping.sort_values("label")["class_name"].astype(str).tolist()

    if split_path.exists():
        split = pd.read_csv(split_path)
        if {"label", "class_name"}.issubset(split.columns):
            return (
                split[["label", "class_name"]]
                .drop_duplicates()
                .sort_values("label")["class_name"]
                .astype(str)
                .tolist()
            )

    return confusion_df.index.astype(str).tolist()


def load_confusion_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    cm = pd.read_csv(path, index_col=0)
    cm.index = cm.index.astype(str)
    cm.columns = cm.columns.astype(str)

    class_order = load_class_order(cm)
    missing_rows = sorted(set(class_order) - set(cm.index))
    missing_cols = sorted(set(class_order) - set(cm.columns))
    if missing_rows or missing_cols:
        raise ValueError(
            "class order does not match confusion CSV; "
            f"missing rows={missing_rows}, missing cols={missing_cols}"
        )

    sources_used.append(path)
    return cm.loc[class_order, class_order].astype(float), class_order


def row_normalize(values: np.ndarray) -> np.ndarray:
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0) * 100.0


def configure_matrix_axes(ax: plt.Axes, class_order: list[str]) -> None:
    labels = [short_label(class_name) for class_name in class_order]
    positions = np.arange(len(class_order))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax.set_yticklabels(labels, rotation=0)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_xlabel("Predicted class", labelpad=12)
    ax.set_ylabel("True class", labelpad=12)

    ax.set_xticks(np.arange(-0.5, len(class_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_aspect("equal")


def annotate_diagonal(ax: plt.Axes, values: np.ndarray, *, normalized: bool) -> None:
    for idx, value in enumerate(np.diag(values)):
        if value <= 0:
            continue
        text = f"{value:.0f}" if normalized else f"{int(round(value))}"
        color = "white" if value >= (60.0 if normalized else 250.0) else "#202020"
        ax.text(idx, idx, text, ha="center", va="center", fontsize=7.5, color=color)


def annotate_count_off_diagonal(ax: plt.Axes, values: np.ndarray, threshold: int = 15) -> None:
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if i == j:
                continue
            value = int(round(values[i, j]))
            if value < threshold:
                continue
            ax.text(j, i, str(value), ha="center", va="center", fontsize=6.8, color="#202020")


def plot_normalized_matrix(values: np.ndarray, class_order: list[str], spec: ExperimentSpec) -> None:
    normalized = row_normalize(values)
    fig, ax = plt.subplots(figsize=(24, 20))
    image = ax.imshow(normalized, cmap="Blues", norm=Normalize(vmin=0, vmax=100), interpolation="nearest")
    configure_matrix_axes(ax, class_order)
    annotate_diagonal(ax, normalized, normalized=True)
    ax.set_title(f"{spec.experiment_id} confusion matrix (row-normalized)", pad=16, fontsize=18)

    cbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.018)
    cbar.set_label("Percent of true class", fontsize=13)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    stem = f"fig_confusion_matrix_{spec.slug}_main"
    save_figure(fig, stem)


def plot_count_matrix(values: np.ndarray, class_order: list[str], spec: ExperimentSpec) -> None:
    fig, ax = plt.subplots(figsize=(24, 20))
    image = ax.imshow(
        values,
        cmap="YlGnBu",
        norm=PowerNorm(gamma=0.42, vmin=0, vmax=max(1.0, float(values.max()))),
        interpolation="nearest",
    )
    configure_matrix_axes(ax, class_order)
    annotate_diagonal(ax, values, normalized=False)
    annotate_count_off_diagonal(ax, values, threshold=15)
    ax.set_title(f"{spec.experiment_id} confusion matrix (counts)", pad=16, fontsize=18)

    cbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.018)
    cbar.set_label("Samples (power-scaled color)", fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    save_figure(fig, f"fig_confusion_matrix_{spec.slug}_counts")


def top_confusion_pairs(values: np.ndarray, class_order: list[str], top_k: int = 15) -> pd.DataFrame:
    row_sums = values.sum(axis=1)
    rows = []
    for true_idx, true_class in enumerate(class_order):
        for pred_idx, predicted_class in enumerate(class_order):
            if true_idx == pred_idx:
                continue
            count = int(values[true_idx, pred_idx])
            if count == 0:
                continue
            percent = 100.0 * count / row_sums[true_idx] if row_sums[true_idx] else 0.0
            rows.append(
                {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "count": count,
                    "percent": percent,
                }
            )
    return pd.DataFrame(rows).sort_values(["count", "percent"], ascending=False).head(top_k)


def plot_top_confusions(values: np.ndarray, class_order: list[str], spec: ExperimentSpec) -> None:
    top = top_confusion_pairs(values, class_order)
    if top.empty:
        skip(f"fig_top_confusions_{spec.slug}", "no off-diagonal confusion entries found")
        return

    plot_df = top.sort_values("count", ascending=False).reset_index(drop=True)
    labels = [display_pair(row.true_class, row.predicted_class) for row in plot_df.itertuples()]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.barh(y, plot_df["count"], color="#4C78A8", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassified test images", fontsize=10)
    ax.set_title(f"Top off-diagonal confusions for {spec.experiment_id}", pad=8, fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(
        bars,
        labels=[f"{row.count} ({row.percent:.1f}%)" for row in plot_df.itertuples()],
        padding=4,
        fontsize=7.5,
    )
    ax.set_xlim(0, max(plot_df["count"]) * 1.22)
    fig.tight_layout()
    save_figure(fig, f"fig_top_confusions_{spec.slug}")


def plot_per_class_recall(values: np.ndarray, class_order: list[str], spec: ExperimentSpec) -> None:
    normalized = row_normalize(values)
    recalls = np.diag(normalized)
    df = pd.DataFrame({"class_name": class_order, "recall": recalls}).sort_values("recall", ascending=True)
    labels = [short_label(class_name) for class_name in df["class_name"]]
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 9.4))
    colors = np.where(df["recall"].to_numpy() < 90.0, "#E45756", "#4C78A8")
    bars = ax.barh(y, df["recall"], color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("Per-class recall (%)", fontsize=10)
    ax.set_title(f"Per-class recall for {spec.experiment_id}", pad=8, fontsize=12)
    ax.set_xlim(max(70.0, float(df["recall"].min()) - 3.0), 100.0)
    ax.grid(axis="x", alpha=0.25)
    mean_recall = float(df["recall"].mean())
    ax.axvline(mean_recall, color="#333333", linewidth=1.2, linestyle="--", label=f"Mean {mean_recall:.1f}%")
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in df["recall"]], padding=3, fontsize=6.6)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"fig_per_class_recall_{spec.slug}")


def generate_for_experiment(spec: ExperimentSpec) -> None:
    try:
        cm, class_order = load_confusion_matrix(spec.confusion_csv)
    except (FileNotFoundError, ValueError) as exc:
        skip(spec.experiment_id, str(exc))
        return

    values = cm.to_numpy(dtype=float)
    print(f"\nexperiment: {spec.experiment_id}")
    print(f"source_confusion_matrix: {spec.confusion_csv}")
    print(f"class_count: {len(class_order)}")
    print(f"test_samples: {int(values.sum())}")
    print(f"diagonal_correct: {int(np.trace(values))}")

    plot_normalized_matrix(values, class_order, spec)
    plot_count_matrix(values, class_order, spec)
    plot_top_confusions(values, class_order, spec)
    plot_per_class_recall(values, class_order, spec)


def main() -> None:
    print(f"output_dir: {FIGURE_DIR}")
    for spec in EXPERIMENTS:
        generate_for_experiment(spec)

    print("\nSummary")
    print("=======")
    print("source artifacts:")
    for path in sources_used:
        print(f"- {path}")
    print(f"\ngenerated files: {len(generated_files)}")
    for path in generated_files:
        print(f"- {path}")
    if skipped:
        print("\nSkipped:")
        for stem, reason in skipped:
            print(f"- {stem}: {reason}")
    else:
        print("\nSkipped: none")


if __name__ == "__main__":
    main()
