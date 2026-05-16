# Phase 9.1: Trace / Debug Utilities
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


def freq_encoding_stats(n_freqs, input_dim, output_dim):
    """Report frequency encoding dimensions."""
    print(f"  Encoding: {input_dim}D → {output_dim}D ({n_freqs} frequencies)")
    return {"n_freqs": n_freqs, "input_dim": input_dim, "output_dim": output_dim}
