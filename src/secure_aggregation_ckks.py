from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch


def _require_tenseal():
    try:
        import tenseal as ts
    except ImportError as exc:
        raise ImportError(
            "TenSEAL is required for CKKS secure aggregation. "
            "Install it with: python -m pip install tenseal"
        ) from exc
    return ts


def tenseal_available() -> bool:
    try:
        import tenseal  # noqa: F401
    except ImportError:
        return False
    return True


def create_ckks_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: list[int] | None = None,
    global_scale: float = 2**40,
):
    """Create a TenSEAL CKKS context for toy secure aggregation tests."""
    ts = _require_tenseal()
    coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale
    return context


def encrypt_vector(vector: Sequence[float] | np.ndarray, context):
    ts = _require_tenseal()
    return ts.ckks_vector(context, np.asarray(vector, dtype=np.float64).tolist())


def decrypt_vector(encrypted_vector) -> np.ndarray:
    return np.asarray(encrypted_vector.decrypt(), dtype=np.float64)


def plaintext_weighted_average(
    vectors: Sequence[Sequence[float] | np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    if len(vectors) != len(weights):
        raise ValueError("vectors and weights must have the same length.")
    if not vectors:
        raise ValueError("At least one vector is required.")

    weights_array = np.asarray(weights, dtype=np.float64)
    if not np.isclose(weights_array.sum(), 1.0):
        weights_array = weights_array / weights_array.sum()

    stacked = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
    return np.sum(stacked * weights_array[:, None], axis=0)


def ckks_weighted_average(encrypted_vectors: Sequence, weights: Sequence[float]):
    if len(encrypted_vectors) != len(weights):
        raise ValueError("encrypted_vectors and weights must have the same length.")
    if not encrypted_vectors:
        raise ValueError("At least one encrypted vector is required.")

    weights_array = np.asarray(weights, dtype=np.float64)
    if not np.isclose(weights_array.sum(), 1.0):
        weights_array = weights_array / weights_array.sum()

    encrypted_sum = encrypted_vectors[0] * float(weights_array[0])
    for encrypted_vector, weight in zip(encrypted_vectors[1:], weights_array[1:]):
        encrypted_sum += encrypted_vector * float(weight)
    return encrypted_sum


def compute_aggregation_error(
    plaintext_avg: Sequence[float] | np.ndarray,
    ckks_avg: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    plaintext = np.asarray(plaintext_avg, dtype=np.float64)
    ckks = np.asarray(ckks_avg, dtype=np.float64)
    if plaintext.shape != ckks.shape:
        raise ValueError(f"Shape mismatch: plaintext={plaintext.shape}, ckks={ckks.shape}")

    absolute_error = np.abs(plaintext - ckks)
    return {
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error)),
    }


def get_classification_head_keys(model_state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Identify timm ViT classifier-head tensors for selected-layer CKKS.

    For timm ViT-Base this is usually `head.weight` and `head.bias`.
    The fallback patterns keep the function usable if a classifier alias is used.
    """
    preferred_keys = ["head.weight", "head.bias"]
    selected = [key for key in preferred_keys if key in model_state_dict]
    if selected:
        return selected

    fallback_prefixes = ("head.", "classifier.", "fc.")
    return [
        key
        for key, tensor in model_state_dict.items()
        if torch.is_floating_point(tensor) and key.startswith(fallback_prefixes)
    ]


def flatten_tensors(
    tensor_dict: dict[str, torch.Tensor],
) -> tuple[np.ndarray, list[dict]]:
    flat_parts = []
    metadata = []

    for key, tensor in tensor_dict.items():
        cpu_tensor = tensor.detach().cpu()
        flat_tensor = cpu_tensor.reshape(-1).to(torch.float64)
        flat_parts.append(flat_tensor.numpy())
        metadata.append(
            {
                "key": key,
                "shape": tuple(cpu_tensor.shape),
                "numel": int(cpu_tensor.numel()),
                "dtype": cpu_tensor.dtype,
            }
        )

    if not flat_parts:
        raise ValueError("No tensors were provided for flattening.")

    return np.concatenate(flat_parts).astype(np.float64), metadata


def unflatten_tensors(
    flat_vector: Sequence[float] | np.ndarray,
    metadata: list[dict],
) -> dict[str, torch.Tensor]:
    vector = np.asarray(flat_vector, dtype=np.float64)
    restored: dict[str, torch.Tensor] = {}
    offset = 0

    for item in metadata:
        numel = int(item["numel"])
        shape = tuple(item["shape"])
        dtype = item["dtype"]
        chunk = vector[offset : offset + numel]
        if len(chunk) != numel:
            raise ValueError("Flat vector is shorter than metadata requires.")
        restored[item["key"]] = torch.tensor(chunk, dtype=dtype).reshape(shape)
        offset += numel

    if offset != len(vector):
        raise ValueError("Flat vector has unused values after unflattening.")
    return restored


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.ndim != 1 or len(weights_array) == 0:
        raise ValueError("weights must be a non-empty 1D sequence.")
    total = weights_array.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return weights_array / total


def plaintext_aggregate_state_dicts(
    client_state_dicts: Sequence[dict[str, torch.Tensor]],
    client_weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    if len(client_state_dicts) != len(client_weights):
        raise ValueError("client_state_dicts and client_weights must have the same length.")
    if not client_state_dicts:
        raise ValueError("At least one client state dict is required.")

    weights = normalize_weights(client_weights)
    first_state = client_state_dicts[0]
    aggregated: dict[str, torch.Tensor] = {}

    for key, first_tensor in first_state.items():
        if torch.is_floating_point(first_tensor):
            weighted_sum = torch.zeros_like(first_tensor.detach().cpu())
            for state, weight in zip(client_state_dicts, weights):
                weighted_sum += state[key].detach().cpu() * float(weight)
            aggregated[key] = weighted_sum
        else:
            aggregated[key] = first_tensor.detach().cpu().clone()

    return aggregated


def ckks_aggregate_selected_state_dicts(
    client_state_dicts: Sequence[dict[str, torch.Tensor]],
    client_weights: Sequence[float],
    selected_keys: Sequence[str],
    context,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int | list[str]]]:
    if not selected_keys:
        raise ValueError("selected_keys must not be empty for selected-layer CKKS.")

    weights = normalize_weights(client_weights)
    selected_keys = list(selected_keys)
    selected_client_tensors = [
        {key: state[key] for key in selected_keys}
        for state in client_state_dicts
    ]

    flat_vectors = []
    metadata = None
    for tensor_dict in selected_client_tensors:
        flat_vector, current_metadata = flatten_tensors(tensor_dict)
        flat_vectors.append(flat_vector)
        if metadata is None:
            metadata = current_metadata

    plaintext_avg = plaintext_weighted_average(flat_vectors, weights)

    encryption_start = time.perf_counter()
    encrypted_vectors = [encrypt_vector(vector, context) for vector in flat_vectors]
    encryption_time = time.perf_counter() - encryption_start

    aggregation_start = time.perf_counter()
    encrypted_avg = ckks_weighted_average(encrypted_vectors, weights)
    aggregation_time = time.perf_counter() - aggregation_start

    decryption_start = time.perf_counter()
    decrypted_avg = decrypt_vector(encrypted_avg)
    decryption_time = time.perf_counter() - decryption_start

    restored_tensors = unflatten_tensors(decrypted_avg, metadata or [])
    error_metrics = compute_aggregation_error(plaintext_avg, decrypted_avg)
    info = {
        "selected_keys": selected_keys,
        "selected_num_parameters": int(len(flat_vectors[0])),
        "ckks_encryption_time": encryption_time,
        "ckks_aggregation_time": aggregation_time,
        "ckks_decryption_time": decryption_time,
        **error_metrics,
    }
    return restored_tensors, info


def mixed_plaintext_ckks_fedavg(
    client_state_dicts: Sequence[dict[str, torch.Tensor]],
    client_weights: Sequence[float],
    selected_keys: Sequence[str],
    context,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int | list[str]]]:
    aggregated = plaintext_aggregate_state_dicts(client_state_dicts, client_weights)
    selected_tensors, ckks_info = ckks_aggregate_selected_state_dicts(
        client_state_dicts=client_state_dicts,
        client_weights=client_weights,
        selected_keys=selected_keys,
        context=context,
    )
    aggregated.update(selected_tensors)
    return aggregated, ckks_info
