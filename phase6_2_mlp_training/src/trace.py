# Phase 6.2: Trace / Debug Utilities
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


def network_stats(network):
    """Report network parameter stats. Returns dict."""
    w = network.layer.weights.to_numpy()
    b = network.layer.biases.to_numpy()
    return {"|W|_mean": float(np.abs(w).mean()), "|b|_mean": float(np.abs(b).mean()),
            "W_shape": str(w.shape)}
