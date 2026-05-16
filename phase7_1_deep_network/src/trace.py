# Phase 7.1: Trace / Debug Utilities
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


def deep_network_stats(network):
    """Report deep network training progress."""
    layers = [network.layer0, network.layer1, network.layer2]
    for i, l in enumerate(layers):
        w = l.weights.to_numpy()
        print(f"  L{i}({l.inputs}→{l.outputs}): |W|={np.abs(w).mean():.3f}")
