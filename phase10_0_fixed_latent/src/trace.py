# Phase 10.0: Trace / Debug Utilities
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


def latent_stats(latent_tensor):
    """Analyze latent texture: resolution and channel stats."""
    arr = latent_tensor.to_numpy()
    return {"shape": latent_tensor.shape,
            "channels": latent_tensor.shape[-1] if len(latent_tensor.shape) > 2 else 1,
            "mean_abs": float(np.abs(arr).mean())}
