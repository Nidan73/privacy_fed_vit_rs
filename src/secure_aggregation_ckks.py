from __future__ import annotations

from typing import Sequence

import numpy as np


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
