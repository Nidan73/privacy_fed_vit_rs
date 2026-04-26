"""CKKS-based secure aggregation placeholder.

TODO:
- check TenSEAL availability
- define CKKS context creation utilities
- encrypt compatible client model updates
- aggregate encrypted updates
- decrypt aggregate updates for the global FedAvg step

This file intentionally contains no fake cryptographic results.
"""


def tenseal_available() -> bool:
    try:
        import tenseal  # noqa: F401
    except ImportError:
        return False
    return True
