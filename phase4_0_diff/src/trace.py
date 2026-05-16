# Phase 4.0: Trace / Debug Utilities
import numpy as np
import slangpy as spy


def tensor_stats(tensor: spy.Tensor, name: str = "tensor") -> dict:
    arr = tensor.to_numpy()
    return {"name": name, "shape": tensor.shape, "dtype": str(tensor.dtype),
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "mean": float(np.mean(arr))}


def print_stats(tensor: spy.Tensor, name: str = "tensor"):
    stats = tensor_stats(tensor, name)
    print(f"[{stats['name']}] shape={stats['shape']} "
          f"min={stats['min']:.3f} max={stats['max']:.3f} mean={stats['mean']:.3f}")


def error_stats(diff_tensor):
    """Return per-pixel error statistics.
    Returns: dict with mean_error, max_error, pct_above(>0.01)"""
    arr = diff_tensor.to_numpy()
    gray = arr[..., 0]
    return {"mean_error": float(gray.mean()), "max_error": float(gray.max()),
            "pct_above_001": float((gray > 0.01).mean()) * 100}
