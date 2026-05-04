# Privacy-Aware Federated ViT for Remote Sensing Scene Classification

This repository contains the research code for **Privacy-Aware Federated Vision Transformer for Remote Sensing Scene Classification**. The project evaluates ViT-Base in centralized and federated settings on remote sensing scene datasets, with selected-layer CKKS aggregation used for privacy-aware FedAvg experiments.

## Project Scope

The method is intentionally bounded:

- Backbone: `timm` `vit_base_patch16_224`
- Pretraining: `pretrained: true`
- Federated learning: FedAvg
- Privacy mechanism: selected-layer chunked CKKS aggregation
- Default CKKS scope: classifier head only
- Extended CKKS scope: classifier head plus final norm
- Full ViT-Base encryption is not used
- Non-selected ViT parameters use plaintext FedAvg

This implementation is a research simulation. CKKS encryption, aggregation, and decryption are executed inside the experimental process. It should not be described as a production secure-aggregation deployment with separated key ownership or threshold decryption.

## Datasets

| Dataset           | Classes | Images | Split                        |
| ----------------- | ------: | -----: | ---------------------------- |
| UCMerced Land Use |      21 |  2,100 | 70% train, 15% val, 15% test |
| NWPU-RESISC45     |      45 | 31,500 | 20% train, 10% val, 70% test |

NWPU-RESISC45 is the primary benchmark. UCMerced is used as the smaller controlled benchmark.

NWPU split counts:

- Train: 6,300 images, 140 per class
- Validation: 3,150 images, 70 per class
- Test: 22,050 images, 490 per class

## Repository Layout

```text
configs/                 Experiment registry and ablation config
data/splits/             Tracked split CSVs and client partitions
papers/draft/figures/    Generated paper figures
figures/                 ACM-ready exported paper figures
results/metrics/         Metrics, confusion matrices, reports
results/tables/          Aggregated result tables
scripts/                 Figure generation scripts
src/                     Dataset, training, FedAvg, CKKS, checks
```

Raw datasets, checkpoints, cache files, and large generated artifacts are intentionally excluded from Git.

## Environment

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, verify CUDA before running full experiments:

```bash
python src/check_gpu.py
```

The completed runs used CUDA-enabled PyTorch on an NVIDIA RTX 5070 Ti.

## Dataset Preparation

### UCMerced

```bash
python src/download_ucmerced_hf.py --output_dir data/raw/uc_merced
python src/dataset_check.py --data_dir data/raw/uc_merced --output_csv results/metrics/dataset_summary.csv
python src/split_data.py --data_dir data/raw/uc_merced --output_dir data/splits --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 --seed 42
python src/client_partition.py --train_csv data/splits/train.csv --num_clients 3 --output_dir data/splits
python src/sanity_check_splits.py
```

### NWPU-RESISC45

```bash
python src/download_nwpu_hf.py --output_dir data/raw/nwpu_resisc45
python src/dataset_check.py --data_dir data/raw/nwpu_resisc45 --output_csv results/metrics/nwpu_dataset_summary.csv
python src/split_data.py --data_dir data/raw/nwpu_resisc45 --output_dir data/splits/nwpu --train_ratio 0.20 --val_ratio 0.10 --test_ratio 0.70 --seed 42
python src/client_partition.py --train_csv data/splits/nwpu/train.csv --num_clients 3 --output_dir data/splits/nwpu
```

Sanity check the NWPU splits and client partitions:

```bash
python src/sanity_check_splits.py --train_csv data/splits/nwpu/train.csv --val_csv data/splits/nwpu/val.csv --test_csv data/splits/nwpu/test.csv --iid_dir data/splits/nwpu/clients_iid --noniid_dir data/splits/nwpu/clients_noniid --expected_train_per_class 140 --expected_val_per_class 70 --expected_test_per_class 490
```

## Client Partitions

NWPU includes several client distributions for three simulated clients:

| Directory                                    | Meaning                    |
| -------------------------------------------- | -------------------------- |
| `data/splits/nwpu/clients_iid`               | IID split                  |
| `data/splits/nwpu/clients_noniid`            | Mild class-skew split      |
| `data/splits/nwpu/clients_dirichlet_alpha03` | Moderate Dirichlet non-IID |
| `data/splits/nwpu/clients_dirichlet_alpha01` | Severe Dirichlet non-IID   |

Generate the severe Dirichlet split:

```bash
python src/client_partition.py --train_csv data/splits/nwpu/train.csv --num_clients 3 --output_dir data/splits/nwpu --partition_type dirichlet --dirichlet_alpha 0.1 --seed 42
```

Validate it:

```bash
python src/sanity_check_splits.py --train_csv data/splits/nwpu/train.csv --val_csv data/splits/nwpu/val.csv --test_csv data/splits/nwpu/test.csv --iid_dir data/splits/nwpu/clients_iid --noniid_dir data/splits/nwpu/clients_noniid --extra_client_dir data/splits/nwpu/clients_dirichlet_alpha01 --expected_train_per_class 140 --expected_val_per_class 70 --expected_test_per_class 490
```

## Experiment Registry

Experiments are configured in [configs/ablation_plan.yaml](configs/ablation_plan.yaml).

List registered experiments:

```bash
python src/experiment_registry.py --config configs/ablation_plan.yaml
```

Check a config without training:

```bash
python src/run_experiment.py --experiment_id N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu --no_train
```

Run a dry-run pipeline check:

```bash
python src/run_experiment.py --experiment_id N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu --dry_run
```

Use `--output_suffix` whenever changing training settings. The launcher refuses training overrides without a suffix to avoid overwriting prior results.

## Main NWPU Experiments

Paper-facing NWPU experiment IDs:

| ID                                              | Setting                    | Privacy        |
| ----------------------------------------------- | -------------------------- | -------------- |
| `N0_centralized_vit_base_nwpu`                  | Centralized ViT-Base       | None           |
| `N1_fedavg_iid_vit_base_nwpu`                   | FedAvg IID                 | None           |
| `N2_fedavg_noniid_vit_base_nwpu`                | FedAvg mild class-skew     | None           |
| `N2D_fedavg_dirichlet03_vit_base_nwpu`          | FedAvg Dirichlet alpha=0.3 | None           |
| `N2D01_fedavg_dirichlet01_vit_base_nwpu`        | FedAvg Dirichlet alpha=0.1 | None           |
| `N3_fedavg_ckks_iid_vit_base_nwpu`              | FedAvg IID                 | Head-only CKKS |
| `N3HN_fedavg_ckks_iid_vit_base_nwpu`            | FedAvg IID                 | Head+Norm CKKS |
| `N4D01_fedavg_ckks_dirichlet01_vit_base_nwpu`   | FedAvg Dirichlet alpha=0.1 | Head-only CKKS |
| `N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu` | FedAvg Dirichlet alpha=0.1 | Head+Norm CKKS |

Example severe plaintext FedAvg run:

```bash
python src/run_experiment.py --experiment_id N2D01_fedavg_dirichlet01_vit_base_nwpu --global_rounds 20 --local_epochs 1 --lr 5e-5 --aug_policy remote_sensing_strong --label_smoothing 0.1 --scheduler cosine --seed 42 --output_suffix 20r1e_lr5e5_aug_ls_cosine_s42
```

Example severe Head+Norm CKKS run:

```bash
python src/run_experiment.py --experiment_id N4D01HN_fedavg_ckks_dirichlet01_vit_base_nwpu --global_rounds 20 --local_epochs 1 --lr 5e-5 --aug_policy remote_sensing_strong --label_smoothing 0.1 --scheduler cosine --seed 42 --output_suffix 20r1e_lr5e5_aug_ls_cosine_s42
```

## CKKS Aggregation

The CKKS implementation lives in [src/secure_aggregation_ckks.py](src/secure_aggregation_ckks.py) and is called from [src/train_fedavg.py](src/train_fedavg.py).

Default selected keys:

```yaml
selected_ckks_keys:
  - head.weight
  - head.bias
```

Extended Head+Norm selected keys:

```yaml
selected_ckks_keys:
  - head.weight
  - head.bias
  - norm.weight
  - norm.bias
```

NWPU encrypted parameter counts:

| Scope     | Keys                                                   | Parameters | CKKS chunks |
| --------- | ------------------------------------------------------ | ---------: | ----------: |
| Head-only | `head.weight`, `head.bias`                             |     34,605 |           9 |
| Head+Norm | `head.weight`, `head.bias`, `norm.weight`, `norm.bias` |     36,141 |           9 |

UCMerced head-only CKKS uses 16,149 selected parameters and 4 chunks.

## Results and Tables

Generated metrics and reports are stored under:

```text
results/metrics/
results/logs/
results/tables/
```

Aggregate result tables:

```bash
python src/aggregate_results.py
python src/compare_experiments.py
```

Important generated table files include:

- `results/tables/experiment_summary.csv`
- `results/tables/experiment_mean_std_summary.csv`
- `results/tables/nwpu_iid_client_class_distribution.csv`
- `results/tables/nwpu_dirichlet_alpha01_client_class_distribution.csv`

## Paper Figures

Generate the main paper figures:

```bash
python scripts/generate_paper_figures.py
```

Generate ACM-ready confusion-matrix figures from saved confusion CSVs:

```bash
python scripts/generate_confusion_figures.py --mode acm_counts
```

Figure outputs are written to:

```text
papers/draft/figures/
figures/
```

The figure scripts do not import training modules, do not load checkpoints, and do not run inference.

## Safety Notes

- Do not commit raw data under `data/raw/`.
- Do not commit checkpoints, `.pt` files, cache folders, or downloaded images.
- Do not overwrite completed experiment outputs.
- Use `--dry_run` or `--no_train` before launching long experiments.
- Use `--output_suffix` for every new full run.
- Test metrics are computed from the best validation checkpoint, not the last checkpoint.

## License and Citation

This repository is part of a research project. If reusing the code, cite the project paper once publication details are available.
