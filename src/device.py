"""
src/device.py
Auto-detects the best compute device.
Mac M-series → MPS  |  NVIDIA → CUDA  |  Fallback → CPU
"""
import platform
import torch
from loguru import logger


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def device_info() -> dict:
    d = get_device()
    info = {"device": str(d), "os": platform.system(), "torch": torch.__version__}
    if d.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"]  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    elif d.type == "mps":
        info["gpu_name"] = "Apple Silicon (MPS)"
    return info


DEVICE = get_device()
