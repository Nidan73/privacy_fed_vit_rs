from __future__ import annotations

import platform
import sys
from importlib import metadata


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def print_possible_causes() -> None:
    print("\nCUDA is not available to PyTorch.")
    print("Possible causes:")
    print("1. CPU-only PyTorch installed")
    print("2. NVIDIA driver not installed or outdated")
    print("3. wrong Python environment selected")
    print("4. CUDA wheel mismatch")
    print("5. RTX 50-series may require a recent PyTorch CUDA build")


def main() -> None:
    print("GPU / CUDA Diagnostic")
    print("=====================")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.replace(chr(10), ' ')}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except ImportError:
        print("torch version: not installed")
        print(f"torchvision version: {package_version('torchvision')}")
        print(f"timm version: {package_version('timm')}")
        print_possible_causes()
        return

    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {package_version('torchvision')}")
    print(f"timm version: {package_version('timm')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception as exc:
        cudnn_version = f"unavailable ({exc})"
    print(f"torch.backends.cudnn.version(): {cudnn_version}")

    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")

    if torch.cuda.is_available():
        for device_index in range(device_count):
            props = torch.cuda.get_device_properties(device_index)
            print(f"GPU {device_index} name: {props.name}")
            print(f"GPU {device_index} capability: {props.major}.{props.minor}")
            print(f"GPU {device_index} total memory: {props.total_memory / (1024 ** 3):.2f} GB")

        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        print(f"Current device name: {torch.cuda.get_device_name(current_device)}")
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated(current_device) / (1024 ** 2):.2f} MB")
        amp_can_be_used = True
        print(f"AMP can be used: {amp_can_be_used}")

        try:
            tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            result = tensor * 2.0
            torch.cuda.synchronize()
            print(f"Tiny CUDA tensor test result: {result.detach().cpu().tolist()}")
        except Exception as exc:
            print(f"Tiny CUDA tensor test failed: {exc}")
    else:
        print("Current device: unavailable")
        print("Allocated GPU memory: unavailable")
        print("AMP can be used: false")
        print("Tiny CUDA tensor test result: skipped because CUDA is not available")
        print_possible_causes()


if __name__ == "__main__":
    main()
