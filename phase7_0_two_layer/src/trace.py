# Phase 7.0: Trace / Debug Utilities
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


def layer_stats(network):
    """Print layer-by-layer parameter stats."""
    for i, layer in enumerate([network.layer0, network.layer1]):
        w = layer.weights.to_numpy()
        b = layer.biases.to_numpy()
        print(f"  Layer {i}: W({layer.outputs}×{layer.inputs}) "
              f"|W|={np.abs(w).mean():.3f} |b|={np.abs(b).mean():.3f}")
