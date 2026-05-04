from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch


DEFAULT_SELECTED_CKKS_KEYS = ("head.weight", "head.bias")


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


def get_ckks_slot_count(poly_modulus_degree: int) -> int:
    if poly_modulus_degree <= 0:
        raise ValueError("poly_modulus_degree must be positive.")
    return poly_modulus_degree // 2


def chunk_vector(vector: Sequence[float] | np.ndarray, chunk_size: int) -> list[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    return [
        array[start : start + chunk_size]
        for start in range(0, len(array), chunk_size)
    ]


def unchunk_vector(chunks: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([np.asarray(chunk, dtype=np.float64).reshape(-1) for chunk in chunks])


def encrypt_vector_chunks(
    vector: Sequence[float] | np.ndarray,
    context,
    chunk_size: int,
) -> list:
    return [encrypt_vector(chunk, context) for chunk in chunk_vector(vector, chunk_size)]


def decrypt_vector_chunks(encrypted_chunks: Sequence) -> np.ndarray:
    return unchunk_vector([decrypt_vector(encrypted_chunk) for encrypted_chunk in encrypted_chunks])


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


def ckks_weighted_average_chunked(
    client_vectors: Sequence[Sequence[float] | np.ndarray],
    weights: Sequence[float],
    context,
    chunk_size: int,
    plaintext_avg: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    if len(client_vectors) != len(weights):
        raise ValueError("client_vectors and weights must have the same length.")
    if not client_vectors:
        raise ValueError("At least one client vector is required.")

    weights_array = normalize_weights(weights)
    vector_lengths = [len(np.asarray(vector).reshape(-1)) for vector in client_vectors]
    if len(set(vector_lengths)) != 1:
        raise ValueError(f"All client vectors must have the same length. Got: {vector_lengths}")

    encryption_start = time.perf_counter()
    encrypted_client_chunks = [
        encrypt_vector_chunks(vector, context, chunk_size)
        for vector in client_vectors
    ]
    encryption_time = time.perf_counter() - encryption_start

    num_chunks = len(encrypted_client_chunks[0])
    if any(len(chunks) != num_chunks for chunks in encrypted_client_chunks):
        raise ValueError("All encrypted client vectors must have the same number of chunks.")

    aggregation_start = time.perf_counter()
    aggregated_encrypted_chunks = []
    for chunk_index in range(num_chunks):
        encrypted_chunk_sum = encrypted_client_chunks[0][chunk_index] * float(weights_array[0])
        for client_chunks, weight in zip(encrypted_client_chunks[1:], weights_array[1:]):
            encrypted_chunk_sum += client_chunks[chunk_index] * float(weight)
        aggregated_encrypted_chunks.append(encrypted_chunk_sum)
    aggregation_time = time.perf_counter() - aggregation_start

    decryption_start = time.perf_counter()
    aggregated_vector = decrypt_vector_chunks(aggregated_encrypted_chunks)
    decryption_time = time.perf_counter() - decryption_start

    max_abs_error = None
    mean_abs_error = None
    if plaintext_avg is not None:
        error_metrics = compute_aggregation_error(plaintext_avg, aggregated_vector)
        max_abs_error = error_metrics["max_absolute_error"]
        mean_abs_error = error_metrics["mean_absolute_error"]

    info = {
        "ckks_encryption_time": encryption_time,
        "ckks_aggregation_time": aggregation_time,
        "ckks_decryption_time": decryption_time,
        "max_absolute_error": max_abs_error,
        "mean_absolute_error": mean_abs_error,
        "num_chunks": num_chunks,
        "chunk_size": int(chunk_size),
    }
    return aggregated_vector, info


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
    preferred_keys = list(DEFAULT_SELECTED_CKKS_KEYS)
    selected = [key for key in preferred_keys if key in model_state_dict]
    if selected:
        return selected

    fallback_prefixes = ("head.", "classifier.", "fc.")
    return [
        key
        for key, tensor in model_state_dict.items()
        if torch.is_floating_point(tensor) and key.startswith(fallback_prefixes)
    ]


def resolve_selected_ckks_keys(
    model_state_dict: dict[str, torch.Tensor],
    requested_keys: Sequence[str] | str | None = None,
) -> list[str]:
    """Resolve and validate selected-layer CKKS keys.

    With no explicit config, this keeps the original head-only behavior. When
    keys are provided, every selected tensor must exist and be floating-point so
    it can be flattened, CKKS-aggregated, and restored safely.
    """
    if requested_keys is None:
        selected = get_classification_head_keys(model_state_dict)
        if not selected:
            raise ValueError("No classifier head keys found for selected-layer CKKS aggregation.")
        return selected

    if isinstance(requested_keys, str):
        selected = [key.strip() for key in requested_keys.split(",") if key.strip()]
    else:
        selected = [str(key).strip() for key in requested_keys if str(key).strip()]

    if not selected:
        raise ValueError("selected_ckks_keys was provided but no valid keys were listed.")

    duplicated = sorted({key for key in selected if selected.count(key) > 1})
    if duplicated:
        raise ValueError(f"selected_ckks_keys contains duplicate keys: {duplicated}")

    missing = [key for key in selected if key not in model_state_dict]
    if missing:
        raise ValueError(f"selected_ckks_keys not found in model state_dict: {missing}")

    non_floating = [key for key in selected if not torch.is_floating_point(model_state_dict[key])]
    if non_floating:
        raise ValueError(f"selected_ckks_keys must be floating-point tensors: {non_floating}")

    return selected


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
    chunk_size: int,
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

    decrypted_avg, chunked_info = ckks_weighted_average_chunked(
        client_vectors=flat_vectors,
        weights=weights,
        context=context,
        chunk_size=chunk_size,
        plaintext_avg=plaintext_avg,
    )

    restored_tensors = unflatten_tensors(decrypted_avg, metadata or [])
    info = {
        "selected_keys": selected_keys,
        "selected_num_parameters": int(len(flat_vectors[0])),
        "ckks_encryption_time": chunked_info["ckks_encryption_time"],
        "ckks_aggregation_time": chunked_info["ckks_aggregation_time"],
        "ckks_decryption_time": chunked_info["ckks_decryption_time"],
        "max_absolute_error": chunked_info["max_absolute_error"],
        "mean_absolute_error": chunked_info["mean_absolute_error"],
        "ckks_num_chunks": int(chunked_info["num_chunks"]),
        "ckks_chunk_size": int(chunked_info["chunk_size"]),
    }
    return restored_tensors, info


def mixed_plaintext_ckks_fedavg(
    client_state_dicts: Sequence[dict[str, torch.Tensor]],
    client_weights: Sequence[float],
    selected_keys: Sequence[str],
    context,
    chunk_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int | list[str]]]:
    aggregated = plaintext_aggregate_state_dicts(client_state_dicts, client_weights)
    selected_tensors, ckks_info = ckks_aggregate_selected_state_dicts(
        client_state_dicts=client_state_dicts,
        client_weights=client_weights,
        selected_keys=selected_keys,
        context=context,
        chunk_size=chunk_size,
    )
    aggregated.update(selected_tensors)
    return aggregated, ckks_info
