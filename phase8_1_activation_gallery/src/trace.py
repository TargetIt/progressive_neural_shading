# Phase 8.1: Trace / Debug Utilities
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


def activation_gallery_stats(outputs):
    """Compare different activation outputs."""
    for name, tensor in outputs.items():
        arr = tensor.to_numpy()
        print(f"  {name}: dead={float((arr==0).mean())*100:.1f}% "
              f"active_mean={float(arr[arr>0].mean()) if (arr>0).any() else 0:.3f}")
