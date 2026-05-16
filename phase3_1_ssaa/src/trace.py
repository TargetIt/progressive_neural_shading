# Phase 1.0: Trace / Debug Utilities
# ====================================
# 提供 shader 输出的基本统计: min, max, mean
# 帮助理解"shader 到底输出了什么"

import numpy as np
import slangpy as spy


def tensor_stats(tensor: spy.Tensor, name: str = "tensor") -> dict:
    """返回 GPU Tensor 的统计信息。

    Returns:
        dict with min, max, mean, shape, dtype
    """
    arr = tensor.to_numpy()
    return {
        "name": name,
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def print_stats(tensor: spy.Tensor, name: str = "tensor"):
    """打印 Tensor 统计信息。"""
    stats = tensor_stats(tensor, name)
    print(f"[{stats['name']}] shape={stats['shape']} "
          f"min={stats['min']:.3f} max={stats['max']:.3f} "
          f"mean={stats['mean']:.3f}")


def verify_solid_color(tensor: spy.Tensor, expected_rgb=(1.0, 0.0, 0.0)):
    """验证所有像素是否为预期颜色。

    Returns:
        (bool, str): 是否通过, 错误信息
    """
    arr = tensor.to_numpy()
    r, g, b = expected_rgb
    tol = 0.01

    if abs(np.mean(arr[..., 0]) - r) > tol:
        return False, f"R channel: expected {r}, got {np.mean(arr[..., 0]):.3f}"
    if abs(np.mean(arr[..., 1]) - g) > tol:
        return False, f"G channel: expected {g}, got {np.mean(arr[..., 1]):.3f}"
    if abs(np.mean(arr[..., 2]) - b) > tol:
        return False, f"B channel: expected {b}, got {np.mean(arr[..., 2]):.3f}"
    return True, "OK"
