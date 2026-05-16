# Phase 10.1: Trace / Debug Utilities
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


def trainable_latent_stats(latent, latent_grad):
    """Report trainable latent texture optimization progress."""
    l = latent.to_numpy(); g = latent_grad.to_numpy()
    return {"|latent|": float(np.abs(l).mean()),
            "|grad|": float(np.abs(g).mean()),
            "shape": latent.shape}
