# Phase 4.2: Trace / Debug Utilities
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


def loss_stats(loss_tensor):
    """Return loss statistics for the error heatmap.
    Returns: dict with mean_loss, max_loss, coverage (pct > 0.001)"""
    arr = loss_tensor.to_numpy()
    gray = (arr[..., 0] + arr[..., 1] + arr[..., 2]) / 3.0
    return {"mean_loss": float(gray.mean()), "max_loss": float(gray.max()),
            "coverage_pct": float((gray > 0.001).mean()) * 100}
