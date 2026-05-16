# Phase 3.1: Trace / Debug Utilities
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


def compare_ssaa(std_1x, ssaa_result):
    """Compare standard 1x render vs SSAA result: verify SSAA reduces aliasing (lower std).
    Returns: (bool, str, float)"""
    arr1 = std_1x.to_numpy(); arr2 = ssaa_result.to_numpy()
    std1, std2 = float(arr1.std()), float(arr2.std())
    diff = float(np.abs(arr1 - arr2).mean())
    ok = diff < 1.0
    return ok, f"std(1x)={std1:.4f} std(SSAA)={std2:.4f} diff={diff:.4f}", diff
