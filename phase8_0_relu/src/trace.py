# Phase 8.0: Trace / Debug Utilities
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


def relu_analysis(tensor):
    """Analyze ReLU output: dead neuron fraction, activation stats."""
    arr = tensor.to_numpy()
    return {"dead_pct": float((arr == 0).mean()) * 100,
            "active_mean": float(arr[arr > 0].mean()) if (arr > 0).any() else 0}
