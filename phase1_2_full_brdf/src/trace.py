# Phase 1.2: Trace / Debug Utilities
# ====================================
# 提供 shader 输出的基本统计和纹理验证。

import numpy as np
import slangpy as spy


def tensor_stats(tensor: spy.Tensor, name: str = "tensor") -> dict:
    """返回 GPU Tensor 的统计信息。"""
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


def verify_texture_output(tensor: spy.Tensor):
    """验证基于纹理的 BRDF 输出: 非零均值 + 像素间有变化。

    Returns:
        (bool, str): 是否通过, 错误信息
    """
    arr = tensor.to_numpy()
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    if mean_val < 0.005:
        return False, f"mean={mean_val:.4f} (输出太暗或被截断)"
    if std_val < 0.005:
        return False, f"std={std_val:.4f} (无纹理变化)"
    return True, f"mean={mean_val:.4f}, std={std_val:.4f}"
