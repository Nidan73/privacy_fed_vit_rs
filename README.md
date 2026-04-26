# Privacy-Aware Federated Vision Transformer for Remote Sensing Scene Classification

This repository contains the code scaffold and dataset preparation utilities for a research project on privacy-aware federated scene classification for remote sensing imagery.

## Frozen Research Setup

- Backbone: ViT-Base
- Federated learning method: FedAvg
- Privacy mechanism: CKKS-based secure aggregation
- Dataset split: 70% train, 15% validation, 15% test
- First dataset: UCMerced Land Use
- Later dataset: NWPU-RESISC45

## Evaluation Plan

1. Centralized ViT-Base
2. FedAvg + ViT-Base with IID clients
3. FedAvg + ViT-Base with non-IID clients
4. FedAvg + ViT-Base + CKKS with IID clients
5. FedAvg + ViT-Base + CKKS with non-IID clients

## Current Completed Status

- Project skeleton created.
- UCMerced Land Use downloaded locally from Hugging Face.
- Raw UCMerced images are excluded from GitHub.
- Dataset checked: 21 classes, 2,100 images, 100 images per class.
- Stratified train/validation/test split created: 1,470 / 315 / 315 images.
- IID and simple class-skewed non-IID client partitions created for 3 clients.

No model training has been started.

## Data Policy

Raw datasets are not committed to GitHub. The repository keeps source code, notebooks, reproducible split CSV files, metric summaries, and paper draft folders. Downloaded images, processed data, model checkpoints, caches, and large generated logs are ignored by `.gitignore`.

Tracked data artifacts:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `data/splits/clients_iid/*.csv`
- `data/splits/clients_noniid/*.csv`
- `results/metrics/dataset_summary.csv`

## Reproduce Current Dataset State

Run these commands from the repository root:

```bash
python src/download_ucmerced_hf.py --output_dir data/raw/uc_merced
python src/dataset_check.py --data_dir data/raw/uc_merced
python src/split_data.py --data_dir data/raw/uc_merced --output_dir data/splits
python src/client_partition.py --train_csv data/splits/train.csv --num_clients 3 --output_dir data/splits
```

## Ablation Plan

Main paper ablations:

- A0: Centralized ViT-Base
- A1: FedAvg + ViT-Base with IID clients
- A2: FedAvg + ViT-Base with non-IID clients
- A3: FedAvg + ViT-Base + CKKS secure aggregation with IID clients
- A4: FedAvg + ViT-Base + CKKS secure aggregation with non-IID clients

Inspect the configured ablations:

```bash
python src/experiment_registry.py --config configs/ablation_plan.yaml
```

Run only a tiny centralized dry-run check:

```bash
python src/run_experiment.py --experiment_id A0_centralized_vit_base --dry_run
```

Run a 10-epoch centralized debug experiment with a separate output suffix:

```bash
python src/run_experiment.py --experiment_id A0_centralized_vit_base --epochs 10 --output_suffix 10ep
```

Use `--output_suffix` when changing experiment settings so previous results are not overwritten. Final test metrics are evaluated using the best validation checkpoint, not the last epoch.

Checkpoint behavior:

- `best.pt` stores the model with the best validation accuracy.
- `last.pt` stores the latest epoch state for recovery, including optimizer state and AMP scaler state when AMP is enabled.
- Checkpoints are ignored by Git.
- Use `--resume_from` to continue an interrupted centralized run.

Example resume command:

```bash
python src/run_experiment.py --experiment_id A0_centralized_vit_base --epochs 10 --output_suffix 10ep_resume --resume_from experiments/centralized/A0_centralized_vit_base_10ep/checkpoints/last.pt
```

A1 FedAvg IID dry-run:

```bash
python src/run_experiment.py --experiment_id A1_fedavg_iid_vit_base --dry_run
```

A1 full small run:

```bash
python src/run_experiment.py --experiment_id A1_fedavg_iid_vit_base --global_rounds 5 --local_epochs 1 --output_suffix 5r1e
```

A1 should be run only after A0 centralized training works. A1 uses the IID client split in `data/splits/clients_iid` and does not include CKKS secure aggregation.

## CKKS Toy Aggregation Test

Validate encrypted weighted averaging before connecting CKKS to FedAvg:

```bash
python src/test_ckks_aggregation.py --vector_length 1024
```

This test uses toy client update vectors and TenSEAL CKKS encryption. Full ViT-Base encryption is not attempted yet because it may be computationally expensive.

## A3 Selected-Layer CKKS FedAvg

A3 uses selected-layer CKKS secure aggregation. Only classifier head parameters, such as `head.weight` and `head.bias`, are CKKS-encrypted during aggregation. The remaining ViT-Base parameters use normal plaintext FedAvg.

This is the practical first privacy-aware setting for this project. Full-model CKKS is not attempted yet because ViT-Base has about 85.8M parameters and full encryption may be too slow and memory-heavy.

Dry-run A3 before any full experiment:

```bash
python src/run_experiment.py --experiment_id A3_fedavg_ckks_iid_vit_base --dry_run
```

## Data Leakage Sanity Check

Run the split and client partition sanity checker before moving to new ablations:

```bash
python src/sanity_check_splits.py
```

Recommended order:

1. Validate centralized ViT-Base first.
2. Implement and validate FedAvg second.
3. Add CKKS secure aggregation third.

Do not run all ablations before each stage is validated.

## GPU Check

Training ViT-Base should be run on a CUDA-capable NVIDIA GPU. Before starting full centralized training, check the active Python and PyTorch CUDA environment:

```bash
python src/check_gpu.py
```

If CUDA is reported as false, do not start full training until the PyTorch CUDA installation, NVIDIA driver, and selected Python environment are fixed.
