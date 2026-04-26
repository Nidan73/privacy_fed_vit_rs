from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from secure_aggregation_ckks import (
    ckks_weighted_average,
    compute_aggregation_error,
    create_ckks_context,
    decrypt_vector,
    encrypt_vector,
    plaintext_weighted_average,
)
from utils import resolve_path


DEFAULT_SAMPLE_COUNTS = [504, 483, 483]
DEFAULT_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]


def build_fake_client_vectors(vector_length: int) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    base_update = rng.normal(loc=0.0, scale=0.05, size=vector_length)
    return [
        base_update + rng.normal(loc=0.01 * client_id, scale=0.01, size=vector_length)
        for client_id in range(3)
    ]


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy CKKS secure aggregation test.")
    parser.add_argument("--vector_length", type=int, default=1024)
    parser.add_argument("--poly_modulus_degree", type=int, default=8192)
    parser.add_argument(
        "--global_scale",
        type=int,
        default=40,
        help="CKKS scale exponent. 40 means global_scale=2**40.",
    )
    parser.add_argument("--output_path", default="results/metrics/ckks_toy_aggregation_metrics.json")
    args = parser.parse_args()

    if args.vector_length < 1:
        raise ValueError("vector_length must be positive.")

    sample_counts = np.asarray(DEFAULT_SAMPLE_COUNTS, dtype=np.float64)
    weights = (sample_counts / sample_counts.sum()).tolist()
    vectors = build_fake_client_vectors(args.vector_length)
    plaintext_avg = plaintext_weighted_average(vectors, weights)
    global_scale = 2 ** int(args.global_scale)

    print("CKKS Toy Secure Aggregation")
    print("===========================")
    print(f"vector_length: {args.vector_length}")
    print(f"sample_counts: {DEFAULT_SAMPLE_COUNTS}")
    print(f"weights: {[round(weight, 6) for weight in weights]}")
    print(f"poly_modulus_degree: {args.poly_modulus_degree}")
    print(f"coeff_mod_bit_sizes: {DEFAULT_COEFF_MOD_BIT_SIZES}")
    print(f"global_scale: 2**{args.global_scale}")

    try:
        context = create_ckks_context(
            poly_modulus_degree=args.poly_modulus_degree,
            coeff_mod_bit_sizes=DEFAULT_COEFF_MOD_BIT_SIZES,
            global_scale=global_scale,
        )
    except ImportError as exc:
        print(f"\nTenSEAL unavailable: {exc}")
        print("If TenSEAL has no wheel for this Python version, use Python 3.10 or 3.11 in a venv/conda env.")
        print("A simulation mode can be added later for timing-only ablations, but this test requires real TenSEAL.")
        raise SystemExit(1) from exc

    encryption_start = time.perf_counter()
    encrypted_vectors = [encrypt_vector(vector, context) for vector in vectors]
    encryption_time = time.perf_counter() - encryption_start

    aggregation_start = time.perf_counter()
    encrypted_average = ckks_weighted_average(encrypted_vectors, weights)
    aggregation_time = time.perf_counter() - aggregation_start

    decryption_start = time.perf_counter()
    ckks_avg = decrypt_vector(encrypted_average)
    decryption_time = time.perf_counter() - decryption_start

    error_metrics = compute_aggregation_error(plaintext_avg, ckks_avg)
    metrics = {
        "tenseal_available": True,
        "vector_length": args.vector_length,
        "num_clients": len(vectors),
        "sample_counts": DEFAULT_SAMPLE_COUNTS,
        "weights": weights,
        "poly_modulus_degree": args.poly_modulus_degree,
        "coeff_mod_bit_sizes": DEFAULT_COEFF_MOD_BIT_SIZES,
        "global_scale_exponent": int(args.global_scale),
        "global_scale": float(global_scale),
        "encryption_time_seconds": encryption_time,
        "aggregation_time_seconds": aggregation_time,
        "decryption_time_seconds": decryption_time,
        **error_metrics,
    }

    output_path = resolve_path(args.output_path)
    save_metrics(metrics, output_path)

    print("\nCKKS Toy Aggregation Results")
    print("============================")
    print(f"max_absolute_error: {metrics['max_absolute_error']:.8e}")
    print(f"mean_absolute_error: {metrics['mean_absolute_error']:.8e}")
    print(f"encryption_time_seconds: {encryption_time:.4f}")
    print(f"aggregation_time_seconds: {aggregation_time:.4f}")
    print(f"decryption_time_seconds: {decryption_time:.4f}")
    print(f"metrics_path: {output_path}")


if __name__ == "__main__":
    main()
