from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
LOGS_DIR = ROOT / "results" / "logs"
SPLITS_DIR = ROOT / "data" / "splits"
FIGURE_DIR = ROOT / "papers" / "draft" / "figures"

PNG_DPI = 300
DEFAULT_FONT_SIZE = 8


plt.rcParams.update(
    {
        "font.size": DEFAULT_FONT_SIZE,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)


generated_files: list[Path] = []
skipped_figures: list[tuple[str, str]] = []


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output_path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
        generated_files.append(output_path)
        print(f"generated: {output_path}")
    plt.close(fig)


def skip_figure(stem: str, reason: str) -> None:
    skipped_figures.append((stem, reason))
    print(f"skipped: {stem} ({reason})")


def pct(value: float) -> float:
    return float(value) * 100.0


def metric_or_none(filename: str) -> dict | None:
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    return load_json(path)


def metric_required(filename: str) -> dict:
    path = METRICS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return load_json(path)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str = "#F7F7F7",
    edgecolor: str = "#333333",
    fontsize: float | None = None,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        linespacing=1.15,
        fontsize=fontsize,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#333333",
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def figure_framework_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    blue = "#DCEBFA"
    green = "#E4F3E0"
    orange = "#FBE7D3"
    gray = "#F4F4F4"
    purple = "#EFE7F7"

    add_box(ax, (0.18, 2.7), 1.15, 0.56, "Dataset\nNWPU/UCM", blue, fontsize=7)
    add_box(ax, (0.18, 1.75), 1.15, 0.56, "Train/val/\ntest split", blue, fontsize=7)
    add_arrow(ax, (0.76, 2.7), (0.76, 2.31))

    client_rows = [(3.35, "Client 1"), (2.35, "Client 2"), (1.35, "Client 3")]
    for y, label in client_rows:
        add_box(ax, (1.82, y), 1.3, 0.5, f"{label}\nlocal train", green, fontsize=6.9)
        add_arrow(ax, (3.12, y + 0.25), (4.35, 2.34), "#333333", rad=0.04 if y > 2.35 else -0.04)

    add_arrow(ax, (1.33, 2.03), (1.82, 2.6), rad=0.02)

    add_box(ax, (4.35, 2.05), 0.95, 0.58, "Client\nupdates", gray, fontsize=7)

    ax.text(5.65, 3.66, "Plaintext backbone path", color="#2A7F62", fontsize=7, fontweight="bold")
    add_box(ax, (5.65, 3.0), 1.1, 0.52, "Backbone\nparams", green, fontsize=7)
    add_box(ax, (7.15, 3.0), 1.1, 0.52, "Weighted\nFedAvg", green, fontsize=7)
    add_arrow(ax, (5.3, 2.47), (5.65, 3.26), "#2A7F62", rad=-0.1)
    add_arrow(ax, (6.75, 3.26), (7.15, 3.26), "#2A7F62")

    ax.text(5.65, 1.68, "Selected-layer CKKS path", color="#C65F21", fontsize=7, fontweight="bold")
    add_box(ax, (5.65, 0.95), 1.1, 0.52, "Select\nhead/norm", orange, fontsize=6.7)
    add_box(ax, (7.15, 0.95), 1.1, 0.52, "Flatten\nchunk", orange, fontsize=6.7)
    add_box(ax, (8.65, 0.95), 1.1, 0.52, "CKKS\nencrypt", orange, fontsize=6.7)
    add_box(ax, (10.15, 0.9), 1.1, 0.62, "Encrypted\nweighted\navg", orange, fontsize=6.4)
    add_box(ax, (11.65, 0.95), 1.1, 0.52, "Decrypt\nreshape", orange, fontsize=6.7)
    add_arrow(ax, (5.3, 2.18), (5.65, 1.21), "#C65F21", rad=0.1)
    add_arrow(ax, (6.75, 1.21), (7.15, 1.21), "#C65F21")
    add_arrow(ax, (8.25, 1.21), (8.65, 1.21), "#C65F21")
    add_arrow(ax, (9.75, 1.21), (10.15, 1.21), "#C65F21")
    add_arrow(ax, (11.25, 1.21), (11.65, 1.21), "#C65F21")

    add_box(ax, (12.15, 2.18), 1.0, 0.58, "Global\nViT", purple)
    add_box(ax, (12.15, 3.35), 1.0, 0.52, "Test\neval", purple)
    add_arrow(ax, (8.25, 3.26), (12.15, 2.55), "#2A7F62", rad=-0.08)
    add_arrow(ax, (12.75, 1.47), (12.65, 2.18), "#C65F21", rad=-0.08)
    add_arrow(ax, (12.65, 2.76), (12.65, 3.35), "#333333")

    legend_handles = [
        Line2D([0], [0], color="#2A7F62", lw=2, label="Plaintext FedAvg"),
        Line2D([0], [0], color="#C65F21", lw=2, label="Chunked CKKS"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, ncol=2)
    ax.set_title("Selected-layer CKKS federated ViT framework", pad=3)
    save_figure(fig, "fig_framework_overview")


def figure_dataset_split_distribution() -> None:
    datasets = ["UCMerced", "NWPU-RESISC45"]
    splits = ["Train", "Validation", "Test"]
    counts = np.array(
        [
            [1470, 315, 315],
            [6300, 3150, 22050],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(datasets))
    width = 0.23
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    for idx, split in enumerate(splits):
        bars = ax.bar(x + (idx - 1) * width, counts[:, idx], width, label=split, color=colors[idx])
        ax.bar_label(bars, labels=[f"{int(v):,}" for v in counts[:, idx]], padding=2, fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_yscale("log")
    ax.set_yticks([100, 300, 1000, 3000, 10000, 30000])
    ax.set_yticklabels(["100", "300", "1k", "3k", "10k", "30k"])
    ax.set_ylabel("Number of images (log scale)")
    ax.set_title("Dataset split distribution")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(100, 36000)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "fig_dataset_split_distribution")


def class_order_from_train() -> list[str]:
    train_csv = SPLITS_DIR / "nwpu" / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    train_df = pd.read_csv(train_csv)
    return (
        train_df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")["class_name"]
        .astype(str)
        .tolist()
    )


def client_matrix(client_dir: Path, class_order: list[str]) -> np.ndarray:
    rows = []
    for client_id in range(3):
        path = client_dir / f"client_{client_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        counts = df["class_name"].astype(str).value_counts()
        rows.append([int(counts.get(class_name, 0)) for class_name in class_order])
    return np.asarray(rows, dtype=float)


def client_summary_rows(matrix: np.ndarray) -> list[list[str]]:
    rows = []
    num_classes = matrix.shape[1]
    for client_id, counts in enumerate(matrix):
        nonzero = counts[counts > 0]
        per_class_range = "0"
        if len(nonzero) > 0:
            per_class_range = f"{int(nonzero.min())}-{int(nonzero.max())}"
        present = int((counts > 0).sum())
        rows.append(
            [
                f"Client {client_id}",
                f"{int(counts.sum()):,}",
                str(present),
                str(num_classes - present),
                per_class_range,
            ]
        )
    return rows


def figure_client_partition_summary() -> None:
    try:
        class_order = class_order_from_train()
        iid = client_matrix(SPLITS_DIR / "nwpu" / "clients_iid", class_order)
        alpha01 = client_matrix(SPLITS_DIR / "nwpu" / "clients_dirichlet_alpha01", class_order)
    except FileNotFoundError as exc:
        skip_figure("fig_client_partition_summary", f"missing client partition CSV: {exc}")
        return

    fig = plt.figure(figsize=(7.2, 3.15), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55])
    ax_iid = fig.add_subplot(grid[0, 0])
    ax_dir = fig.add_subplot(grid[0, 1])

    ax_iid.axis("off")
    ax_iid.set_title("IID partition\nnear-uniform by design", pad=5, fontsize=8.5)
    table = ax_iid.table(
        cellText=client_summary_rows(iid),
        colLabels=["Client", "Samples", "Classes", "Missing", "Per class"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    table.scale(1.0, 1.28)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if row == 0:
            cell.set_facecolor("#E4F3E0")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FAFAFA")
    ax_iid.text(
        0.5,
        0.08,
        "All clients contain all 45 classes.\nEach class contributes 46-47 train images/client.",
        transform=ax_iid.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#333333",
    )

    dir_rows = client_summary_rows(alpha01)
    dir_samples = ", ".join(row[1] for row in dir_rows)
    dir_classes = ", ".join(row[2] for row in dir_rows)
    dir_missing = ", ".join(row[3] for row in dir_rows)
    image = ax_dir.imshow(alpha01, aspect="auto", cmap="YlGnBu", vmin=0, vmax=float(alpha01.max()))
    ax_dir.set_title(
        "Dirichlet alpha=0.1 label skew\n"
        f"samples: {dir_samples} | classes: {dir_classes} | missing: {dir_missing}",
        pad=5,
        fontsize=8.0,
    )
    ax_dir.set_xlabel("Class index (sorted by label)")
    ax_dir.set_xticks(np.arange(0, len(class_order), 5))
    ax_dir.set_xticklabels([str(i + 1) for i in range(0, len(class_order), 5)])
    ax_dir.set_yticks([0, 1, 2])
    ax_dir.set_yticklabels(["Client 0", "Client 1", "Client 2"])
    ax_dir.set_ylabel("Client")
    cbar = fig.colorbar(image, ax=ax_dir, shrink=0.88, pad=0.018)
    cbar.set_label("Samples per class")
    save_figure(fig, "fig_client_partition_summary")


NWPU_METRIC_FILES = [
    ("N0\ncentral", "results/metrics/N0_centralized_vit_base_nwpu_20ep_lr5e5_aug_ls_cosine_metrics.json", "Centralized"),
    ("N1\nIID", "results/metrics/N1_fedavg_iid_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json", "Plain FedAvg"),
    ("N2\nmild", "results/metrics/N2_fedavg_noniid_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json", "Plain FedAvg"),
    ("N2D\nalpha=.3", "results/metrics/N2D_fedavg_dirichlet03_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json", "Plain FedAvg"),
    ("N2D01\ns42", "results/metrics/N2D01_fedavg_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json", "Plain FedAvg"),
    ("N3\nCKKS", "results/metrics/N3_fedavg_ckks_iid_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json", "CKKS"),
    ("N3HN\nCKKS", "results/metrics/N3HN_fedavg_ckks_iid_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json", "CKKS"),
    ("N4D01\nCKKS", "results/metrics/N4D01_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json", "CKKS"),
    ("N4D01HN\ns42", "results/metrics/N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json", "CKKS"),
    ("N4D01HN\ns123", "results/metrics/N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s123_metrics.json", "CKKS"),
]


def figure_nwpu_experiment_comparison() -> None:
    labels = []
    accuracies = []
    groups = []
    missing = []
    for label, rel_path, group in NWPU_METRIC_FILES:
        path = ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        data = load_json(path)
        if data.get("test_accuracy") is None:
            missing.append(f"{rel_path} (test_accuracy missing)")
            continue
        labels.append(label)
        accuracies.append(pct(data["test_accuracy"]))
        groups.append(group)

    if missing:
        skip_figure("fig_nwpu_experiment_comparison", f"missing required metrics: {missing}")
        return

    colors = {"Centralized": "#8F8F8F", "Plain FedAvg": "#4C78A8", "CKKS": "#F58518"}
    display_labels = [label.replace("\n", " ") for label in labels]
    y = np.arange(len(display_labels))[::-1]
    bar_colors = [colors[group] for group in groups]
    fig, ax = plt.subplots(figsize=(5.6, 3.65))
    bars = ax.barh(y, accuracies, color=bar_colors, height=0.62)
    for bar, value in zip(bars, accuracies):
        ax.text(value + 0.025, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("NWPU-RESISC45 experiment comparison", pad=22)
    ax.set_xlim(93.4, 95.45)
    ax.grid(axis="x", alpha=0.25)
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=7, label=label)
        for label, color in colors.items()
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout()
    save_figure(fig, "fig_nwpu_experiment_comparison")


def load_metric_values(filenames: list[str], metric_key: str) -> np.ndarray:
    values = []
    for filename in filenames:
        data = metric_required(filename)
        value = data.get(metric_key)
        if value is None:
            raise ValueError(f"{filename} missing {metric_key}")
        values.append(pct(value))
    return np.asarray(values, dtype=float)


def figure_two_seed_stability() -> None:
    try:
        plain_acc = load_metric_values(
            [
                "N2D01_fedavg_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json",
                "N2D01_fedavg_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s123_metrics.json",
            ],
            "test_accuracy",
        )
        plain_f1 = load_metric_values(
            [
                "N2D01_fedavg_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json",
                "N2D01_fedavg_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s123_metrics.json",
            ],
            "test_macro_f1",
        )
        ckks_acc = load_metric_values(
            [
                "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json",
                "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s123_metrics.json",
            ],
            "test_accuracy",
        )
        ckks_f1 = load_metric_values(
            [
                "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json",
                "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s123_metrics.json",
            ],
            "test_macro_f1",
        )
    except (FileNotFoundError, ValueError) as exc:
        skip_figure("fig_two_seed_stability", str(exc))
        return

    means = np.array(
        [
            [plain_acc.mean(), plain_f1.mean()],
            [ckks_acc.mean(), ckks_f1.mean()],
        ]
    )
    stds = np.array(
        [
            [plain_acc.std(ddof=1), plain_f1.std(ddof=1)],
            [ckks_acc.std(ddof=1), ckks_f1.std(ddof=1)],
        ]
    )

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    x = np.arange(2)
    width = 0.34
    bars_plain = ax.bar(x - width / 2, means[0], width, yerr=stds[0], capsize=3, label="Plain FedAvg", color="#4C78A8")
    bars_ckks = ax.bar(x + width / 2, means[1], width, yerr=stds[1], capsize=3, label="Head+Norm CKKS", color="#F58518")
    for bars, values, errors in ((bars_plain, means[0], stds[0]), (bars_ckks, means[1], stds[1])):
        for bar, value, error in zip(bars, values, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + 0.035,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1"])
    ax.set_ylabel("Mean test score (%)")
    ax.set_title("Two-seed stability under Dirichlet alpha=0.1")
    ax.set_ylim(93.15, 94.62)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.02, "Error bars: sample std over seeds 42 and 123", ha="center", fontsize=6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "fig_two_seed_stability")


def first_existing_metric(filenames: list[str]) -> dict | None:
    for filename in filenames:
        data = metric_or_none(filename)
        if data is not None:
            return data
    return None


def figure_ckks_scope_overhead() -> None:
    ucm_head = first_existing_metric(
        [
            "A3_fedavg_ckks_iid_vit_base_10r1e_chunked_s42_metrics.json",
            "A3_fedavg_ckks_iid_vit_base_10r1e_chunked_metrics.json",
        ]
    )
    nwpu_head = metric_or_none("N3_fedavg_ckks_iid_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json")
    nwpu_head_norm = metric_or_none("N3HN_fedavg_ckks_iid_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json")
    if not all([ucm_head, nwpu_head, nwpu_head_norm]):
        skip_figure("fig_ckks_scope_overhead", "required CKKS metrics missing")
        return

    scope_labels = ["UCM\nHead", "NWPU\nHead", "NWPU\nHead+Norm"]
    params = [
        int(ucm_head["selected_ckks_num_parameters"]),
        int(nwpu_head["selected_ckks_num_parameters"]),
        int(nwpu_head_norm["selected_ckks_num_parameters"]),
    ]
    chunks = [
        int(ucm_head["ckks_num_chunks"]),
        int(nwpu_head["ckks_num_chunks"]),
        int(nwpu_head_norm["ckks_num_chunks"]),
    ]

    timing_specs = [
        ("N3\nHead", "N3_fedavg_ckks_iid_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json"),
        ("N3HN\nHead+Norm", "N3HN_fedavg_ckks_iid_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json"),
        ("N4D01\nHead", "N4D01_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json"),
        ("N4D01HN\nHead+Norm", "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_metrics.json"),
    ]
    timing_rows = []
    for label, filename in timing_specs:
        data = metric_or_none(filename)
        if not data:
            continue
        rounds = int(data.get("actual_global_rounds") or data.get("global_rounds") or 1)
        timing_rows.append(
            {
                "label": label,
                "enc": float(data.get("ckks_encryption_time_total", 0.0)) / rounds,
                "agg": float(data.get("ckks_aggregation_time_total", 0.0)) / rounds,
                "dec": float(data.get("ckks_decryption_time_total", 0.0)) / rounds,
            }
        )

    if timing_rows:
        fig = plt.figure(figsize=(6.4, 4.1), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
        ax_params = fig.add_subplot(grid[0, 0])
        ax_chunks = fig.add_subplot(grid[0, 1])
        ax_time = fig.add_subplot(grid[1, :])
    else:
        fig = plt.figure(figsize=(5.2, 2.8), constrained_layout=True)
        grid = fig.add_gridspec(1, 2)
        ax_params = fig.add_subplot(grid[0, 0])
        ax_chunks = fig.add_subplot(grid[0, 1])
        ax_time = None

    ax_params.bar(scope_labels, params, color=["#8F8F8F", "#F58518", "#E45756"], width=0.65)
    ax_params.set_title("Encrypted parameters")
    ax_params.set_ylabel("Parameters")
    ax_params.bar_label(ax_params.containers[0], labels=[f"{v:,}" for v in params], padding=2, fontsize=6)
    ax_params.grid(axis="y", alpha=0.25)

    ax_chunks.bar(scope_labels, chunks, color=["#8F8F8F", "#F58518", "#E45756"], width=0.65)
    ax_chunks.set_title("CKKS chunks")
    ax_chunks.set_ylabel("Chunks")
    ax_chunks.bar_label(ax_chunks.containers[0], labels=[str(v) for v in chunks], padding=2, fontsize=6)
    ax_chunks.set_ylim(0, max(chunks) + 2)
    ax_chunks.grid(axis="y", alpha=0.25)

    if ax_time is not None:
        y = np.arange(len(timing_rows))
        enc = np.array([row["enc"] for row in timing_rows])
        agg = np.array([row["agg"] for row in timing_rows])
        dec = np.array([row["dec"] for row in timing_rows])
        ax_time.barh(y, enc, label="Encrypt", color="#4C78A8")
        ax_time.barh(y, agg, left=enc, label="Aggregate", color="#72B7B2")
        ax_time.barh(y, dec, left=enc + agg, label="Decrypt", color="#F58518")
        ax_time.set_yticks(y)
        ax_time.set_yticklabels([row["label"].replace("\n", " ") for row in timing_rows])
        ax_time.invert_yaxis()
        ax_time.set_title("Average CKKS time per round")
        ax_time.set_xlabel("Seconds")
        ax_time.grid(axis="x", alpha=0.25)
        ax_time.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=6)

    save_figure(fig, "fig_ckks_scope_overhead")


def figure_confusion_matrix_n4d01hn() -> None:
    path = METRICS_DIR / "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_confusion_matrix.csv"
    if not path.exists():
        skip_figure("fig_confusion_matrix_n4d01hn", f"missing confusion matrix: {path}")
        return
    cm = pd.read_csv(path, index_col=0)
    values = cm.to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    tick_positions = np.arange(0, normalized.shape[0], 5)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(i + 1) for i in tick_positions])
    ax.set_yticklabels([str(i + 1) for i in tick_positions])
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title("N4D01HN confusion matrix (row-normalized, seed 42)")
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("Percent of true class")
    fig.tight_layout()
    save_figure(fig, "fig_confusion_matrix_n4d01hn")


def read_curve(path: Path, x_col: str, y_col: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"{path} missing {x_col} or {y_col}")
    return pd.DataFrame({"step": df[x_col], "val_accuracy": df[y_col] * 100.0, "label": label})


def figure_training_curves_selected() -> None:
    specs = [
        (
            LOGS_DIR / "N0_centralized_vit_base_nwpu_20ep_lr5e5_aug_ls_cosine_epoch_log.csv",
            "epoch",
            "val_accuracy",
            "N0 centralized",
        ),
        (
            LOGS_DIR / "N2D01_fedavg_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_round_log.csv",
            "round",
            "val_accuracy",
            "N2D01 plain alpha=0.1",
        ),
        (
            LOGS_DIR / "N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu_20r1e_lr5e5_aug_ls_cosine_s42_round_log.csv",
            "round",
            "val_accuracy",
            "N4D01HN CKKS alpha=0.1",
        ),
    ]

    curves = []
    missing = []
    for path, x_col, y_col, label in specs:
        if not path.exists():
            missing.append(path.as_posix())
            continue
        try:
            curves.append(read_curve(path, x_col, y_col, label))
        except ValueError as exc:
            missing.append(str(exc))

    if missing:
        skip_figure("fig_training_curves_selected", f"missing logs: {missing}")
        return

    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    colors = ["#8F8F8F", "#4C78A8", "#F58518"]
    for curve, color in zip(curves, colors):
        ax.plot(curve["step"], curve["val_accuracy"], marker="o", markersize=3, linewidth=1.2, label=curve["label"].iloc[0], color=color)
    ax.set_xlabel("Epoch / global round")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Selected validation curves")
    ax.set_ylim(50, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_training_curves_selected")


def main() -> None:
    print(f"output_dir: {FIGURE_DIR}")
    figure_framework_overview()
    figure_dataset_split_distribution()
    figure_client_partition_summary()
    figure_nwpu_experiment_comparison()
    figure_two_seed_stability()
    figure_ckks_scope_overhead()
    figure_confusion_matrix_n4d01hn()
    figure_training_curves_selected()

    print("\nSummary")
    print("=======")
    print(f"generated files: {len(generated_files)}")
    for path in generated_files:
        print(f"- {path}")
    if skipped_figures:
        print("\nSkipped figures:")
        for stem, reason in skipped_figures:
            print(f"- {stem}: {reason}")
    else:
        print("\nSkipped figures: none")


if __name__ == "__main__":
    main()
