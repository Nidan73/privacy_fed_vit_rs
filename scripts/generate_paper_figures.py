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
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    blue = "#DCEBFA"
    green = "#E4F3E0"
    orange = "#FBE7D3"
    gray = "#F4F4F4"
    purple = "#EFE7F7"

    add_box(ax, (0.2, 3.9), 1.45, 0.85, "Dataset\nNWPU / UCM", blue)
    add_box(ax, (0.2, 2.55), 1.45, 0.85, "Train / Val /\nTest split", blue)
    add_arrow(ax, (0.93, 3.9), (0.93, 3.4))

    for y, label in zip((4.6, 3.55, 2.5), ("Client 1", "Client 2", "Client 3")):
        add_box(ax, (2.0, y), 1.25, 0.62, f"{label}\nlocal data", gray)
        add_box(ax, (3.55, y), 1.35, 0.62, "ViT-Base\nlocal train", green)
        add_arrow(ax, (3.25, y + 0.31), (3.55, y + 0.31))
    add_arrow(ax, (1.65, 2.98), (2.0, 2.81))
    add_arrow(ax, (1.65, 2.98), (2.0, 3.86))
    add_arrow(ax, (1.65, 2.98), (2.0, 4.91))

    add_box(ax, (5.35, 4.2), 1.35, 0.75, "Plaintext\nFedAvg path", green)
    add_box(ax, (7.05, 4.2), 1.55, 0.75, "Backbone\nparameters", green)
    add_arrow(ax, (4.9, 4.9), (5.35, 4.58), "#2A7F62", -0.08)
    add_arrow(ax, (4.9, 3.86), (5.35, 4.58), "#2A7F62")
    add_arrow(ax, (4.9, 2.81), (5.35, 4.58), "#2A7F62", 0.08)
    add_arrow(ax, (6.7, 4.58), (7.05, 4.58), "#2A7F62")

    add_box(ax, (5.05, 2.05), 1.35, 0.75, "Select\nhead / norm", orange)
    add_box(ax, (6.75, 2.05), 1.35, 0.75, "Flatten\nand chunk", orange)
    add_box(ax, (8.45, 2.05), 1.3, 0.75, "CKKS\nencrypt", orange)
    add_box(ax, (6.75, 0.85), 1.35, 0.75, "Encrypted\nweighted avg", orange)
    add_box(ax, (8.45, 0.85), 1.3, 0.75, "Decrypt\nreshape", orange)
    add_arrow(ax, (4.9, 4.9), (5.05, 2.42), "#C65F21", 0.18)
    add_arrow(ax, (4.9, 3.86), (5.05, 2.42), "#C65F21")
    add_arrow(ax, (4.9, 2.81), (5.05, 2.42), "#C65F21", -0.08)
    add_arrow(ax, (6.4, 2.42), (6.75, 2.42), "#C65F21")
    add_arrow(ax, (8.1, 2.42), (8.45, 2.42), "#C65F21")
    add_arrow(ax, (9.1, 2.05), (7.42, 1.6), "#C65F21", -0.2)
    add_arrow(ax, (8.1, 1.22), (8.45, 1.22), "#C65F21")

    add_box(ax, (9.65, 3.2), 1.15, 0.8, "Global\nViT model", purple)
    add_box(ax, (9.65, 4.6), 1.15, 0.7, "Test\nevaluation", purple)
    add_arrow(ax, (8.6, 4.58), (9.65, 3.75), "#2A7F62", -0.05)
    add_arrow(ax, (9.75, 1.22), (10.25, 3.2), "#C65F21", -0.12)
    add_arrow(ax, (10.22, 4.0), (10.22, 4.6), "#333333")

    legend_handles = [
        Line2D([0], [0], color="#2A7F62", lw=2, label="Plaintext FedAvg backbone"),
        Line2D([0], [0], color="#C65F21", lw=2, label="Selected-layer CKKS"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False)
    ax.set_title("Proposed selected-layer CKKS federated ViT framework", pad=4)
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

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    x = np.arange(len(datasets))
    width = 0.23
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    for idx, split in enumerate(splits):
        bars = ax.bar(x + (idx - 1) * width, counts[:, idx], width, label=split, color=colors[idx])
        ax.bar_label(bars, labels=[f"{int(v):,}" for v in counts[:, idx]], padding=2, fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset split distribution")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, 24500)
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


def figure_client_partition_summary() -> None:
    try:
        class_order = class_order_from_train()
        iid = client_matrix(SPLITS_DIR / "nwpu" / "clients_iid", class_order)
        alpha01 = client_matrix(SPLITS_DIR / "nwpu" / "clients_dirichlet_alpha01", class_order)
    except FileNotFoundError as exc:
        skip_figure("fig_client_partition_summary", f"missing client partition CSV: {exc}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    matrices = [
        ("IID clients\nclasses/client: 45, 45, 45", iid),
        ("Dirichlet alpha=0.1\nclasses/client: 24, 26, 24", alpha01),
    ]
    vmax = max(float(iid.max()), float(alpha01.max()))
    image = None
    for ax, (title, matrix) in zip(axes, matrices):
        image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Class index (sorted by label)")
        ax.set_xticks(np.arange(0, len(class_order), 5))
        ax.set_xticklabels([str(i + 1) for i in range(0, len(class_order), 5)])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Client 0", "Client 1", "Client 2"])
    axes[0].set_ylabel("Client")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.015)
        cbar.set_label("Samples per class")
    fig.suptitle("NWPU client label distribution: IID vs severe non-IID", y=1.02)
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
    bar_colors = [colors[group] for group in groups]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=bar_colors, width=0.72)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in accuracies], padding=2, fontsize=6)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("NWPU-RESISC45 experiment comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(93.0, 96.0)
    ax.grid(axis="y", alpha=0.25)
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=7, label=label)
        for label, color in colors.items()
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper left")
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
    ax.bar_label(bars_plain, labels=[f"{v:.2f}" for v in means[0]], padding=2, fontsize=6)
    ax.bar_label(bars_ckks, labels=[f"{v:.2f}" for v in means[1]], padding=2, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1"])
    ax.set_ylabel("Mean test score (%)")
    ax.set_title("Two-seed stability under Dirichlet alpha=0.1")
    ax.set_ylim(93.2, 94.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.02, 0.02, "Error bars: sample std over seeds 42 and 123", transform=ax.transAxes, fontsize=6)
    fig.tight_layout()
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
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.0, 0.85, 1.65]})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))

    axes[0].bar(scope_labels, params, color=["#8F8F8F", "#F58518", "#E45756"])
    axes[0].set_title("Encrypted params")
    axes[0].set_ylabel("Parameters")
    axes[0].bar_label(axes[0].containers[0], labels=[f"{v:,}" for v in params], padding=2, fontsize=6)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(scope_labels, chunks, color=["#8F8F8F", "#F58518", "#E45756"])
    axes[1].set_title("CKKS chunks")
    axes[1].set_ylabel("Chunks")
    axes[1].bar_label(axes[1].containers[0], labels=[str(v) for v in chunks], padding=2, fontsize=6)
    axes[1].set_ylim(0, max(chunks) + 2)
    axes[1].grid(axis="y", alpha=0.25)

    if timing_rows:
        ax = axes[2]
        x = np.arange(len(timing_rows))
        enc = np.array([row["enc"] for row in timing_rows])
        agg = np.array([row["agg"] for row in timing_rows])
        dec = np.array([row["dec"] for row in timing_rows])
        ax.bar(x, enc, label="Encrypt", color="#4C78A8")
        ax.bar(x, agg, bottom=enc, label="Aggregate", color="#72B7B2")
        ax.bar(x, dec, bottom=enc + agg, label="Decrypt", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in timing_rows], rotation=30, ha="right")
        ax.set_title("Avg CKKS time / round")
        ax.set_ylabel("Seconds")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=6)

    fig.suptitle("Selected-layer CKKS scope and measured overhead", y=1.05)
    fig.tight_layout()
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
