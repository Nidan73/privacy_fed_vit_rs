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
