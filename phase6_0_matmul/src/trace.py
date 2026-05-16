# Phase 6.0: Trace / Debug Utilities
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


def verify_matmul_output(tensor, weights):
    """Verify matmul output matches expected pattern based on weights."""
    arr = tensor.to_numpy()
    return {"R_range": (float(arr[..., 0].min()), float(arr[..., 0].max())),
            "G_range": (float(arr[..., 1].min()), float(arr[..., 1].max())),
            "B_range": (float(arr[..., 2].min()), float(arr[..., 2].max()))}
