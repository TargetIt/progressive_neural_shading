# Phase 3.0: Trace / Debug Utilities
# ====================================
# 提供 shader 输出的基本统计和高分辨率对比验证。

import numpy as np
import slangpy as spy


def tensor_stats(tensor: spy.Tensor, name: str = "tensor") -> dict:
    arr = tensor.to_numpy()
    return {
        "name": name, "shape": tensor.shape, "dtype": str(tensor.dtype),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def print_stats(tensor: spy.Tensor, name: str = "tensor"):
    stats = tensor_stats(tensor, name)
    print(f"[{stats['name']}] shape={stats['shape']} "
          f"min={stats['min']:.3f} max={stats['max']:.3f} "
          f"mean={stats['mean']:.3f}")


def compare_1x_vs_2x(out_1x, out_2x_down):
    """对比 1x 和超采样后降采样的 2x 输出差异。

    Returns:
        (bool, str, float): pass/fail, message, mean_absolute_diff
    """
    arr1 = out_1x.to_numpy()
    arr2 = out_2x_down.to_numpy()
    diff = np.abs(arr1 - arr2).mean()
    if diff > 1.0:
        return False, f"large diff: {diff:.4f}", diff
    return True, f"diff={diff:.4f}", diff
