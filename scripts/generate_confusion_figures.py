from __future__ import annotations

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
EXPERIMENT_NAME = "N4D01HN"
CONFUSION_CSV = (
    METRICS_DIR
    / "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_confusion_matrix.csv"
)


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 7,
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


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
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
            return (
                mapping.sort_values("label")["class_name"]
                .astype(str)
                .tolist()
            )

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
            f"class order does not match confusion CSV; "
            f"missing rows={missing_rows}, missing cols={missing_cols}"
        )

    return cm.loc[class_order, class_order].astype(float), class_order


def row_normalize(values: np.ndarray) -> np.ndarray:
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0) * 100.0


def annotate_matrix(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    normalized: bool,
    offdiag_threshold: float,
) -> None:
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            is_diagonal = i == j
            if not is_diagonal and value < offdiag_threshold:
                continue
            if value <= 0:
                continue
            if normalized:
                text = f"{value:.0f}"
                color = "white" if value >= 55 else "#1f1f1f"
            else:
                text = f"{int(round(value))}"
                color = "white" if value >= 260 else "#1f1f1f"
            ax.text(j, i, text, ha="center", va="center", fontsize=3.4, color=color)


def plot_confusion_matrix(
    values: np.ndarray,
    class_order: list[str],
    *,
    stem: str,
    title: str,
    normalized: bool,
) -> None:
    labels = [short_label(class_name) for class_name in class_order]
    fig, ax = plt.subplots(figsize=(11.2, 9.8))

    if normalized:
        image = ax.imshow(values, cmap="Blues", norm=Normalize(vmin=0, vmax=100), aspect="equal")
        cbar_label = "Percent of true class"
        annotate_matrix(ax, values, normalized=True, offdiag_threshold=5.0)
    else:
        max_value = max(1.0, float(values.max()))
        image = ax.imshow(values, cmap="YlGnBu", norm=PowerNorm(gamma=0.48, vmin=0, vmax=max_value), aspect="equal")
        cbar_label = "Samples"
        annotate_matrix(ax, values, normalized=False, offdiag_threshold=10.0)

    positions = np.arange(len(class_order))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax.set_yticklabels(labels)
    ax.tick_params(axis="both", length=0, pad=2)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, pad=8)

    ax.set_xticks(np.arange(-0.5, len(class_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.018)
    cbar.set_label(cbar_label)
    if normalized:
        cbar.set_ticks([0, 20, 40, 60, 80, 100])

    fig.tight_layout()
    save_figure(fig, stem)


def top_confusion_pairs(values: np.ndarray, class_order: list[str], top_k: int = 10) -> pd.DataFrame:
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


def plot_top_confusions(values: np.ndarray, class_order: list[str]) -> None:
    top = top_confusion_pairs(values, class_order)
    if top.empty:
        skip("fig_top_confusions_n4d01hn", "no off-diagonal confusion entries found")
        return

    plot_df = top.sort_values("count", ascending=True)
    labels = [display_pair(row.true_class, row.predicted_class) for row in plot_df.itertuples()]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.barh(y, plot_df["count"], color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Misclassified test images")
    ax.set_title(f"Top off-diagonal confusions for {EXPERIMENT_NAME}", pad=8)
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(
        bars,
        labels=[f"{row.count} ({row.percent:.1f}%)" for row in plot_df.itertuples()],
        padding=3,
        fontsize=6,
    )
    ax.set_xlim(0, max(plot_df["count"]) * 1.22)
    fig.tight_layout()
    save_figure(fig, "fig_top_confusions_n4d01hn")


def plot_per_class_accuracy(normalized: np.ndarray, class_order: list[str]) -> None:
    recalls = np.diag(normalized)
    df = pd.DataFrame({"class_name": class_order, "recall": recalls}).sort_values("recall", ascending=True)
    labels = [short_label(class_name) for class_name in df["class_name"]]
    y = np.arange(len(df))

    fig_height = max(7.0, len(df) * 0.17)
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    colors = np.where(df["recall"].to_numpy() < 90.0, "#E45756", "#4C78A8")
    ax.barh(y, df["recall"], color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("Per-class recall (%)")
    ax.set_title(f"Per-class accuracy for {EXPERIMENT_NAME}", pad=8)
    ax.set_xlim(max(70.0, float(df["recall"].min()) - 3.0), 100.0)
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(float(df["recall"].mean()), color="#333333", linewidth=0.9, linestyle="--", label="Mean")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig_per_class_accuracy_n4d01hn")


def main() -> None:
    print(f"source_confusion_matrix: {CONFUSION_CSV}")
    try:
        cm, class_order = load_confusion_matrix(CONFUSION_CSV)
    except (FileNotFoundError, ValueError) as exc:
        skip("N4D01HN confusion figures", str(exc))
        return

    values = cm.to_numpy(dtype=float)
    normalized = row_normalize(values)
    print(f"class_count: {len(class_order)}")
    print(f"test_samples: {int(values.sum())}")
    print(f"diagonal_correct: {int(np.trace(values))}")

    plot_confusion_matrix(
        normalized,
        class_order,
        stem="fig_confusion_matrix_n4d01hn_normalized",
        title=f"{EXPERIMENT_NAME} confusion matrix (row-normalized)",
        normalized=True,
    )
    plot_confusion_matrix(
        values,
        class_order,
        stem="fig_confusion_matrix_n4d01hn_counts",
        title=f"{EXPERIMENT_NAME} confusion matrix (counts)",
        normalized=False,
    )
    plot_top_confusions(values, class_order)
    plot_per_class_accuracy(normalized, class_order)

    print("\nSummary")
    print("=======")
    print(f"generated files: {len(generated_files)}")
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
