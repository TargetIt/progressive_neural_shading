# Phase 5.1: Trace / Debug Utilities
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


def ad_gradient_stats(albedo_grad, normal_grad, roughness_grad):
    """Summarize AD-computed gradients.
    Returns: dict with |grad| per parameter"""
    ag = albedo_grad.to_numpy(); ng = normal_grad.to_numpy(); rg = roughness_grad.to_numpy()
    return {"|grad_albedo|": float(np.abs(ag).mean()),
            "|grad_normal|": float(np.abs(ng).mean()),
            "|grad_roughness|": float(np.abs(rg).mean())}
